
#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <perfcounter.h>
#include <stdio.h>
#include <defs.h>
#include <barrier.h>
#include <string.h>
#include "../../PIM-common/common/include/common.h"
#include "emb_types.h"

BARRIER_INIT(mybarrier, NR_TASKLETS);
STDOUT_BUFFER_INIT(1048576)
__mram_noinit int32_t table[MEGABYTE(14)];
__mram_noinit uint32_t input_indices[off_length*pooling_factor*2];
__mram_noinit uint32_t input_offsets[off_length*2];
__mram_noinit int32_t results[10*off_length*NR_COLS/Col_DPU];
__mram_noinit struct query_len len;

__host struct query_len len_wram;
__host int b_idx[pooling_factor*NR_TASKLETS];
__host int offset[off_length+1];
__host int offset_length;
__host uint32_t nb_cycles;
int batch;
int off_set,off_seto;
int per_dpu;
int lookup(int batch_size,T* local_buffer){
    int base;
    int each_tasklet;
    T* local_buffer2 = (T*)mem_alloc(per_dpu*sizeof(T));
    if(batch_size%NR_TASKLETS!=0)   each_tasklet=(batch_size/NR_TASKLETS)+1;
    else    each_tasklet=batch_size/NR_TASKLETS;
    for(int i=0;i<each_tasklet;i++){
        int index = i+me()*each_tasklet;
        if(index+1<=batch_size && (uint32_t)(index+1+batch)<len_wram.nr_batches){
            int start=offset[index+batch];
            int cur = start;
            int end=offset[1+index+batch];
            int size = end - start;
            memset(local_buffer2,0,per_dpu*sizeof(T));
            for(int j=0;j<size;j++){
                int last_time = offset[batch];
                if(cur-last_time>=0 && (uint32_t)(cur-last_time)<len_wram.indices_len){
                    base = b_idx[cur-last_time];
                    if(base>=0){
                        if(NR_COLS == Col_DPU)  mram_read(&table[ALIGN(base,2)],local_buffer,8);
                        else{
                            int sz = per_dpu;
                            int copied_sz = 0;
                            while(copied_sz<sz){
                                mram_read(&table[ALIGN(base*per_dpu+copied_sz,2)],&local_buffer[copied_sz],ALIGN(MIN(2048,(sz-copied_sz)*sizeof(T)),8));
                                copied_sz += 2048/sizeof(T);
                            }
                        }
                        for(int k=0;k<per_dpu;k++)  local_buffer2[k] += local_buffer[k];
                    }
                    cur +=1;
                }
            }
            mram_write(local_buffer2,&(results[ALIGN(i*per_dpu+me()*each_tasklet*per_dpu+batch*per_dpu,2)]),ALIGN(per_dpu * sizeof(T),8));/////////
        }
    }
    barrier_wait(&mybarrier);
    if(me()==0){
        batch += batch_size;
    }
    mem_reset();
    return 0;
}
int main(){
    perfcounter_config(COUNT_CYCLES, true);
    if(me() == 0) {
        mem_reset();
        len_wram.indices_len = len.indices_len;
        len_wram.nr_batches = len.nr_batches;
        len_wram.max = len.max;
        batch = 0;
        off_set = 0;
        off_seto = 0;
        per_dpu = NR_COLS/Col_DPU;
        offset_length = len_wram.nr_batches;
        int sz = len_wram.nr_batches+1;
        int copied_sz = 0;
        while(copied_sz<sz){
            mram_read(&input_offsets[copied_sz],&offset[copied_sz],ALIGN(MIN(2048,(sz-copied_sz)*sizeof(int)),8));
            copied_sz += 2048/sizeof(int);
        }
    }
    barrier_wait(&mybarrier);

    T* local_buffer = (T*)mem_alloc(ALIGN(per_dpu*sizeof(T),8));
    int loop = offset_length/NR_TASKLETS;

    for(int i=0;i<loop;i++){
        if(me()==0){
            off_set=offset[NR_TASKLETS*(i+1)-1]-off_seto;
            int sz = off_set;
            int copied_sz=0;
            //////////////mram_read(&input_indices[off_seto],&b_idx[0],ALIGN(MIN(2048,(sz)*sizeof(int)),8));
            while(copied_sz<sz){
                mram_read(&input_indices[off_seto+copied_sz],&b_idx[copied_sz],ALIGN(MIN(2048,(sz-copied_sz)*sizeof(int)),8));
                copied_sz += 2048/sizeof(int);
            }
            off_seto = offset[NR_TASKLETS*(i+1)];
        }
        barrier_wait(&mybarrier);
        lookup(NR_TASKLETS,local_buffer);
    }
    barrier_wait(&mybarrier);
    if(offset_length%NR_TASKLETS!=0){
        if(me()==0){
            off_set = offset[offset_length-1] - off_seto;
            int sz = off_set;
            ////////////////mram_read(&input_indices[off_seto],&b_idx[0],ALIGN(MIN(2048,(sz)*sizeof(int)),8));
            int copied_sz = 0;
            while(copied_sz<sz){
                mram_read(&input_indices[off_seto+copied_sz],&b_idx[copied_sz],ALIGN(MIN(2048,(sz-copied_sz)*sizeof(int)),8));
                copied_sz += 2048/sizeof(int);
            }
        }
        barrier_wait(&mybarrier);
        lookup(offset_length%NR_TASKLETS,local_buffer);
    }


    nb_cycles = perfcounter_get();
    return 0;
}


