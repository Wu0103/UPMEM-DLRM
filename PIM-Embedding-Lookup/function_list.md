/UPMEM:
    ./run.sh: 
        changed random_env() to make system configureable as showed in README
    /scr/emb_host.c:
        changed populate_mram() function to preload table data to dpu_set_t
        changed lookup() function to first transfer index data and offset data to DPUs, and then invoke kernel program to do embedding lookup in DPUs; After that will get results back to CPU and convert the results into float point data
    /src/dpu/emb_dpu_lookup.c:
        changed main() funcion to load data from MRAM into WRAM
        changed lookup() function to do embedding lookup

