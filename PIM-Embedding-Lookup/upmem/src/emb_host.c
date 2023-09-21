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
    res[table_id] = malloc(MAX_NR_BATCHES*NR_COLS*sizeof(int));

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

gettimeofday(&cpudpus, NULL);
    lengths.indices_len=indices_len;
    lengths.nr_batches=nr_batches;
#pragma omp parallel for
for(int i=0;i<table_id;i++){
    DPU_ASSERT(dpu_broadcast_to(dpu_ranks[i], "input_indices", 0 , &indices[i*indices_len],ALIGN(indices_len*sizeof(uint32_t),8), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(dpu_ranks[i], "input_offsets", 0 , &offsets[i*nr_batches], nr_batches*sizeof(uint32_t), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(dpu_ranks[i], "len", 0 , &lengths, sizeof(struct query_len), DPU_XFER_DEFAULT));
}
#pragma omp barrier
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

gettimeofday(&dpucpue, NULL);
dpucpu = get_runtime(dpucpus.tv_sec, dpucpus.tv_usec, dpucpue.tv_sec, dpucpue.tv_usec);
cpudpu = get_runtime(cpudpus.tv_sec, cpudpus.tv_usec, cpudpue.tv_sec, cpudpue.tv_usec);
upmem = get_runtime(upmems.tv_sec, upmems.tv_usec, upmeme.tv_sec, upmeme.tv_usec);
printf("cpu-dpu: %.4f\n",cpudpu);
printf("kernel: %.4f\n",upmem);
printf("dpu-cpu: %.4f\n",dpucpu);

//for(int i=0;i<20;i++){
//DPU_FOREACH(dpu_ranks[i], dpu) DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
//}
//#pragma omp parallel for
//for(int i=0;i<table_id;i++) free(res[i]);
    return 0;

}
int
main() {
}