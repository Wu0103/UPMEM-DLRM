#!/bin/bash
com=./run.sh

function loop(){
    for batch in 1 4 16 64 256 1024
    do
        make clean
        heightdef="export MAX_NR_BATCHES=${batch}"
        sed -i "44c ${heightdef}" ${com}
        ./run.sh -br random
    done
}

loop



