// to compile the code: gcc -O0 -g3 --std=c99 -o emb_host emb_host.c -g `dpu-pkg-config --cflags
// --libs dpu` to build a shared library: gcc -shared -Wl,-soname,emb_host -o emblib.so -fPIC
// emb_host.c `dpu-pkg-config --cflags --libs dpu`
#include "common.h"
#include "host/include/host.h"
#include "emb_types.h"

#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <omp.h>
#define RT_CONFIG 0

#ifndef DPU_BINARY
#    define DPU_BINARY "../upmem/emb_dpu_lookup" // Relative path regarding the PyTorch code
#endif

int32_t* buffer_data[NR_DPUS*NR_TABLES];
int* res[NR_TABLES];


struct dpu_set_t dpu_ranks[AVAILABLE_RANKS];

#define TIME_NOW(_t) (clock_gettime(CLOCK_MONOTONIC, (_t)))

/**
 * @struct dpu_runtime
 * @brief DPU execution times
 */
typedef struct dpu_runtime_totals {
    double execution_time_prepare;
    double execution_time_populate_copy_in;
    double execution_time_copy_in;
    double execution_time_copy_out;
    double execution_time_aggregate_result;
    double execution_time_launch;
} dpu_runtime_totals;


static void enomem() {
    fprintf(stderr, "Out of memory\n");
    exit(ENOMEM);
}


static int alloc_buffers(uint32_t table_id, int32_t *table_data, uint64_t nr_rows) {

    for(int j=0; j<NR_DPUS; j++){

        size_t sz = nr_rows*sizeof(int)*NR_COLS/NR_DPUS;
        buffer_data[j] = malloc(ALIGN(sz,8));
        if (buffer_data[j] == NULL) {
            return ENOMEM;
        }

        for(int k=0; k<nr_rows*NR_COLS/NR_DPUS; k++){
            buffer_data[j][k] = table_data[k*NR_DPUS+j];
        }

    }


    return 0;
}

/*
    Params:
    0. table_id: embedding table number.
    1. nr_rows: number of rows of the embedding table
    2. NR_COLS: number of columns of the embedding table
    3. table_data: a pointer of the size nr_rows*NR_COLS containing table's data
    Result:
    This function breaks down each embedding table into chunks of maximum MAX_CAPACITY
    and pushes each chunk(buffer) to one dpu as well as number of rows and columns of the
    corresponding table with the index of the first and last row held in each dpu.
*/

void populate_mram(uint32_t table_id, uint64_t nr_rows, int32_t *table_data, dpu_runtime_totals *runtime){
    struct timespec start, end;
    if(table_id>=AVAILABLE_RANKS){
        fprintf(stderr,"%d ranks available but tried to load table %dth",AVAILABLE_RANKS,table_id);
        exit(1);
    }
    res[table_id] = malloc(20*ALIGN(NR_DPUS*MAX_NR_BATCHES*NR_COLS*sizeof(int)/COL_DPU,8));

    if (alloc_buffers(table_id, table_data, nr_rows) != 0) {
        enomem();
    }
    struct dpu_set_t set, dpu, dpu_rank;
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
    uint32_t len;
    uint8_t dpu_id,rank_id;

    printf("Transfer Table %d to DPU \n",table_id+1);

    DPU_FOREACH(set, dpu, dpu_id){
        DPU_ASSERT(dpu_prepare_xfer(dpu, buffer_data[dpu_id]));
    }
    DPU_ASSERT(dpu_push_xfer(set,DPU_XFER_TO_DPU, "table", 0, ALIGN(sizeof(int32_t)*NR_COLS*nr_rows/NR_DPUS,8), DPU_XFER_DEFAULT));

    printf("Table %d Transfer Done \n",table_id+1);

    for (int i = 0; i < NR_DPUS; i++){
        free(buffer_data[i]);
    }

    dpu_ranks[table_id] = set;
    return;
}
double
get_runtime(double start_sec, double start_us, double end_sec, double end_us){
    double duration = (end_sec * 1000 + end_us / 1000) - (start_sec * 1000 + start_us / 1000 );
    return duration;
}
struct timeval cpudpus,cpudpue,upmems,upmeme,dpucpus,dpucpue;
double cpudpu,upmem,dpucpu;

int32_t* lookup(uint32_t* indices, uint32_t *offsets, uint64_t indices_len,uint64_t nr_batches, float *final_results, uint32_t table_id){
    //struct timespec start, end;
    int dpu_id;
    uint64_t copied_indices;
    struct dpu_set_t dpu;
    struct query_len lengths;

    cpudpu = 0;
    upmem = 0;
    dpucpu = 0;

    indices_len /= table_id;
    nr_batches /= table_id;
    printf("************************************************************************************ \n");

    struct query_len len[table_id][NR_DPUS/COL_DPU];
    int idx_limit [table_id][NR_DPUS/COL_DPU];
    int record[table_id][NR_DPUS/COL_DPU];
    int offset_dpu [table_id][(MAX_NR_BATCHES+1)][NR_DPUS/COL_DPU];
    typedef struct {
        int b_idx[MAX_NR_BATCHES*POOLING];
        int offset[MAX_NR_BATCHES+1];
        int offset_length;
    } input;
    input *data = (input*)calloc(table_id*NR_DPUS/COL_DPU,sizeof(input));

    gettimeofday(&cpudpus, NULL);
    lengths.indices_len=indices_len;
    lengths.nr_batches=nr_batches;
    int row = NR_COLS/COL_DPU;
    if(row==1){
        #pragma omp parallel for
        for(int i=0;i<table_id;i++){
            DPU_ASSERT(dpu_broadcast_to(dpu_ranks[i], "input_indices", 0 , &indices[i*indices_len],ALIGN(indices_len*sizeof(uint32_t),8), DPU_XFER_DEFAULT));
            DPU_ASSERT(dpu_broadcast_to(dpu_ranks[i], "input_offsets", 0 , &offsets[i*nr_batches], nr_batches*sizeof(uint32_t), DPU_XFER_DEFAULT));
            DPU_ASSERT(dpu_broadcast_to(dpu_ranks[i], "len", 0 , &lengths, sizeof(struct query_len), DPU_XFER_DEFAULT));
        }
        #pragma omp barrier
    }else{
        memset(idx_limit,0,(NR_DPUS/COL_DPU)*sizeof(int)*table_id);
        memset(offset_dpu,0,(NR_DPUS/COL_DPU)*(MAX_NR_BATCHES+1)*sizeof(int)*table_id);
        #pragma omp parallel for
        for(int i=0;i<table_id;i++){
            int rowdpu = (NR_DPUS/COL_DPU);
            int eachdpu = ROW_TOTAL/rowdpu;
            for(int x=0;x<rowdpu;x++){
                data[i*rowdpu+x].offset_length = 1;
                memset(data[i*rowdpu+x].b_idx,0,sizeof(int)*MAX_NR_BATCHES*POOLING);
                memset(data[i*rowdpu+x].offset,0,sizeof(int)*(MAX_NR_BATCHES+1));
            }
            for(int n=0,j = 1;n<indices_len;n++){
                int base = indices[i*indices_len+n]/eachdpu;    
                int off = idx_limit[i][base];
                data[i*rowdpu+base].b_idx[off] = indices[i*indices_len+n]%eachdpu;
                idx_limit[i][base]++;
                offset_dpu[i][j-1][base] = 1;
                data[i*rowdpu+base].offset[data[i*rowdpu+base].offset_length] = idx_limit[i][base];
                if((n==(offsets[i*nr_batches+j]-1))){
                    j++;
                    for(int k=0;k<(rowdpu);k++){
                        if(data[i*rowdpu+k].offset[data[i*rowdpu+k].offset_length]!=0)    data[i*rowdpu+k].offset_length++;
                    }
                }
            }
            
            int max=0;
            for(int x=0;x<rowdpu;x++){
                len[i][x].indices_len = idx_limit[i][x];
                if(max<idx_limit[i][x]) max = idx_limit[i][x];
            }
            DPU_FOREACH(dpu_ranks[i], dpu, dpu_id){DPU_ASSERT(dpu_prepare_xfer(dpu,&data[dpu_id/COL_DPU+i*rowdpu].b_idx[0]));}
            DPU_ASSERT(dpu_push_xfer(dpu_ranks[i], DPU_XFER_TO_DPU, "input_indices",0,ALIGN(max*sizeof(uint32_t),8), DPU_XFER_DEFAULT));
            max=0;
            for(int x=0;x<rowdpu;x++){
                record[i][x] = data[i*rowdpu+x].offset_length;
                if(record[i][x]>=MAX_NR_BATCHES) record[i][x] = MAX_NR_BATCHES;
                len[i][x].nr_batches = record[i][x];
                if(max<record[i][x]) max = record[i][x];
            }
            DPU_FOREACH(dpu_ranks[i], dpu, dpu_id){DPU_ASSERT(dpu_prepare_xfer(dpu,&data[dpu_id/COL_DPU+i*rowdpu].offset[0]));}
            DPU_ASSERT(dpu_push_xfer(dpu_ranks[i], DPU_XFER_TO_DPU, "input_offsets",0,ALIGN(max*sizeof(uint32_t),8), DPU_XFER_DEFAULT));

            DPU_FOREACH(dpu_ranks[i], dpu, dpu_id){DPU_ASSERT(dpu_prepare_xfer(dpu,&len[i][dpu_id/COL_DPU]));}
            DPU_ASSERT(dpu_push_xfer(dpu_ranks[i], DPU_XFER_TO_DPU, "len",0,sizeof(struct query_len), DPU_XFER_DEFAULT));
            
        }
    }
gettimeofday(&cpudpue, NULL);
gettimeofday(&upmems, NULL);
for(int i=0;i<table_id;i++){
    DPU_ASSERT(dpu_launch(dpu_ranks[i], DPU_ASYNCHRONOUS));
}
for(int i=0;i<table_id;i++){
	DPU_ASSERT(dpu_sync(dpu_ranks[i]));
}
gettimeofday(&upmeme, NULL);
gettimeofday(&dpucpus, NULL);

if(row==1){
    #pragma omp parallel for
    for(int i=0;i<table_id;i++){
        DPU_FOREACH(dpu_ranks[i], dpu, dpu_id){DPU_ASSERT(dpu_prepare_xfer(dpu,&res[i][dpu_id*nr_batches*NR_COLS/NR_DPUS]));}
        DPU_ASSERT(dpu_push_xfer(dpu_ranks[i], DPU_XFER_FROM_DPU, "results",0,ALIGN(sizeof(int32_t)*nr_batches*NR_COLS/NR_DPUS,2), DPU_XFER_DEFAULT));
        for (int j=0; j<NR_COLS; j++){
            for(int n=0; n<nr_batches; n++)
                //final_results[n*NR_COLS+j]=(float)res[i][j*nr_batches+n]/1000000000;
                final_results[i*NR_COLS*nr_batches+n*NR_COLS+j]=(float)res[i][j*nr_batches+n]/1000000000;
        }
    }
}else{
    
    int rowdpu = (NR_DPUS/COL_DPU);
    int record_position[rowdpu*table_id];
    memset(record_position,0,sizeof(int)*rowdpu*table_id);
    #pragma omp parallel for
    for(int i=0;i<table_id;i++){
        int max_size = 0;
        int rowdpu = NR_DPUS/COL_DPU;
        for(int n=0;n<rowdpu;n++){
            if(max_size<record[i][n]){
                max_size = record[i][n];
            }
        }
        if(max_size>=MAX_NR_BATCHES) max_size = MAX_NR_BATCHES;
        int size;
        if(ALIGN(sizeof(int32_t)*max_size*NR_COLS/COL_DPU,8)<ALIGN(sizeof(int32_t)*MAX_NR_BATCHES*NR_COLS/COL_DPU,8)) size = ALIGN(sizeof(int32_t)*max_size*NR_COLS/COL_DPU,8);
        else size = ALIGN(sizeof(int32_t)*MAX_NR_BATCHES*NR_COLS/COL_DPU,8);
        DPU_FOREACH(dpu_ranks[i], dpu, dpu_id){DPU_ASSERT(dpu_prepare_xfer(dpu,&res[i][dpu_id*max_size*(NR_COLS/COL_DPU)]));}
        DPU_ASSERT(dpu_push_xfer(dpu_ranks[i], DPU_XFER_FROM_DPU, "results",0,size, DPU_XFER_DEFAULT));
        for(int k=0;k<rowdpu;k++){
            for(int x=0;x<(record[i][k]);x++){
                if(offset_dpu[i][x][k]!=0){
                    for(int m=0;m<COL_DPU;m++){
                        for(int j=0;j<NR_COLS/COL_DPU;j++){
                            final_results[i*NR_COLS*nr_batches+j+m*NR_COLS/COL_DPU+x*NR_COLS] += res[i][record_position[k+i*rowdpu]*(NR_COLS/COL_DPU)+j];
                        }
                    }
                    record_position[k+i*rowdpu]++;
                }   
            }
        }
        for(int n=0; n<max_size; n++){
            for (int j=0; j<NR_COLS; j++){
                final_results[i*NR_COLS*nr_batches+n*NR_COLS+j]=(float)final_results[i*NR_COLS*nr_batches+n*NR_COLS+j]/1000000000;
            }
        }
    }
    
    free(data);
}

gettimeofday(&dpucpue, NULL);
dpucpu = get_runtime(dpucpus.tv_sec, dpucpus.tv_usec, dpucpue.tv_sec, dpucpue.tv_usec);
cpudpu = get_runtime(cpudpus.tv_sec, cpudpus.tv_usec, cpudpue.tv_sec, cpudpue.tv_usec);
upmem = get_runtime(upmems.tv_sec, upmems.tv_usec, upmeme.tv_sec, upmeme.tv_usec);
printf("cpu-dpu: %.4f\n",cpudpu);
printf("kernel: %.4f\n",upmem);
printf("dpu-cpu: %.4f\n",dpucpu);

    return 0;

}
int
main() {
}