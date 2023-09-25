
#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <perfcounter.h>
#include <stdio.h>
#include <defs.h>
#include <barrier.h>
#include <string.h>
#include "common/include/common.h"
#include "emb_types.h"

BARRIER_INIT(mybarrier, NR_TASKLETS);
STDOUT_BUFFER_INIT(1048576)
__mram_noinit int32_t table[MEGABYTE(14)];
__mram_noinit uint32_t input_indices[off_length*pooling_factor];
__mram_noinit uint32_t input_offsets[off_length];
__mram_noinit int32_t results[off_length*NR_COLS/NR_DPUS];
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
                        if(NR_COLS == NR_DPUS)  mram_read(&table[ALIGN(base,2)],local_buffer,8);
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
        per_dpu = NR_COLS/NR_DPUS;
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

#include "common/include/common.h"
#include "emb_types.h"

#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <sem.h>

__mram_noinit struct query_len input_lengths;

__mram_noinit int32_t emb_data[MEGABYTE(14)];
__mram_noinit uint32_t input_indices[32*MAX_NR_BATCHES];
__mram_noinit uint32_t input_offsets[MAX_NR_BATCHES];
__mram_noinit int32_t results[MAX_NR_BATCHES];

uint32_t indices_ptr[NR_TASKLETS];
SEMAPHORE_INIT(first_run_sem,1);
SEMAPHORE_INIT(result_sem,1);

uint32_t indices_len, nr_batches, copied_indices;
__dma_aligned struct query_len lengths;
__dma_aligned uint32_t indices[32*MAX_NR_BATCHES], offsets[MAX_NR_BATCHES];
__dma_aligned int32_t tmp_results[MAX_NR_BATCHES];

__host uint8_t first_run = 1;
int
main() {
    __dma_aligned int32_t read_buff[2];  
    sem_take(&first_run_sem);
    if(first_run==1){
        mem_reset();
        copied_indices=0;

        mram_read(&input_lengths, &lengths, ALIGN(sizeof(struct query_len),8));
        indices_len=lengths.indices_len;
        nr_batches=lengths.nr_batches;

        while(copied_indices<indices_len){
            mram_read(&input_indices[copied_indices],&indices[copied_indices],
            ALIGN(MIN(2048, (indices_len-copied_indices)*sizeof(uint32_t)),8));
            copied_indices+=2048/sizeof(uint32_t);
        }
        mram_read(input_offsets,offsets,ALIGN(nr_batches*sizeof(uint32_t),8));
        first_run=0;
    }
    sem_give(&first_run_sem);

    if(me()!=0)
        indices_ptr[me()]=offsets[me()];
    else
         indices_ptr[me()]=0;

    uint32_t last_written=0;
    for (uint64_t i=me(); i< nr_batches; i+=NR_TASKLETS){

        tmp_results[i]=0;
        while ( (i==nr_batches-1 && indices_ptr[me()]<indices_len) || 
        (i<nr_batches-1 && indices_ptr[me()]<offsets[i+1]) )
        {
            uint32_t ind = indices[indices_ptr[me()]];
            mram_read(&emb_data[ind],read_buff,8);
            tmp_results[i]+=read_buff[((ind % 2) != 0)];
            indices_ptr[me()]++;
        }

        if((i-1)%512==0 || i==nr_batches-1){
            sem_take(&result_sem);
            mram_write(&tmp_results[last_written],&results[last_written], ALIGN(i*sizeof(int32_t),8));
            last_written=i+1;
            sem_give(&result_sem);
        }
        
        if(i+NR_TASKLETS<nr_batches){
            indices_ptr[me()]=offsets[i+NR_TASKLETS];
        }
    }
    sem_take(&first_run_sem);
     if(first_run==0){
         first_run=1;
     }
     sem_give(&first_run_sem);
    return 0;
}
*/