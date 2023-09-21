#!/bin/sh

set -e

build=false
run=false
debug=false
verbose=false

usage() {
    printf "%s %s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n" \
        "USAGE:" "${0}" \
        "[ -b ] - Build code" \
        "[ -r ] - Run code" \
        "[ -d ] - Debug" \
        "[ -V ] - Verbose" \
        "<DATASET - kaggle | random | toy>"
    exit 0
}

die() {
    echo "${1}" 1>&2
    exit 1
}

dataset_valid() {
    [ "${1}" = "kaggle" ] && return 0
    [ "${1}" = "random" ] && return 0
    [ "${1}" = "toy" ] && return 0

    return 1
}

kaggle_env() {
    export NR_TABLES=26
    export NR_COLS=16
    export MAX_NR_BATCHES=512
    export NR_TASKLETS=14
}

random_env() {
export NR_TABLES=4     #
export NR_COLS=93
export MAX_NR_BATCHES=16
export NR_TASKLETS=16
export NR_DPUS=93
export TABLE_ROW=1000000-1000000-1000000-1000000
export ROW_TOTAL=1000000
#export COL_DPU=93
export POOLING=20
pooling="#define pooling_factor ${POOLING}"
sed -i "6c ${pooling}" /home/kaist_icn/wuxiangyu/upload/dlrm/PIM-Embedding-Lookup/upmem/PIM-common/common/include/common.h
rowtotal="#define row_total ${ROW_TOTAL}"
sed -i "9c ${rowtotal}" /home/kaist_icn/wuxiangyu/upload/dlrm/PIM-Embedding-Lookup/upmem/PIM-common/common/include/common.h
}

toy_env() {
    export NR_COLS=8
    export DPU_TEST=1
    export NR_TASKLETS=1
}

global_env() {
    "${verbose}" && export V=1
    "${debug}" && export DEBUG=1
    return 0
}

random_run() {
    python3 "${cwd}/../PIM-dlrm-new/dlrm_dpu_pytorch.py" \
           --arch-embedding-size=${TABLE_ROW} \
           --arch-sparse-feature-size=${NR_COLS} \
           --arch-mlp-bot=1440-720-${NR_COLS} \
           --arch-mlp-top=40-20-10-1 \
           --data-generation=random \
           --mini-batch-size=${MAX_NR_BATCHES} \
           --num-batches=50 \
           --num-indices-per-lookup=${POOLING} \
           --num-indices-per-lookup-fixed=True \
           --inference-only
}

toy_run() {
    echo "DPU_TEST=${DPU_TEST}"
    python3 "${cwd}/c_test.py"
}

build_code() {
    # Set ENV
    "${dataset}_env"
    global_env

    make
}
kaggle_run() {
    dlrm="${cwd}/../dlrm"
    python "${dlrm}/dlrm_dpu_pytorch.py" \
           --arch-sparse-feature-size=16 \
           --arch-mlp-bot="13-512-256-64-16" \
           --arch-mlp-top="512-256-1" \
           --data-generation=dataset \
           --data-set=kaggle \
           --processed-data-file="${dlrm}/raw_data/kaggleAdDisplayChallenge_processed.npz" \
           --load-model="${dlrm}/trainedModels/kaggle-model-graham-final.pt" \
           --mini-batch-size=5 \
           --nepochs=1 \
           --inference-only
}
run_code() {
    cwd="${PWD}"
    if "${debug}"; then
        build_dir="./build/debug/"
    else
        build_dir="./build/release/"
    fi
    cd "${build_dir}"
    "${dataset}_run"
}

main() {
    make clean
    "${build}" && build_code
    "${run}" && run_code
}

options=':bVrdh'
while getopts "${options}" option; do
    case $option in
        b  ) build=true;;
        r  ) run=true;;
        d  ) debug=true;;
        V  ) verbose=true;;
        h  ) usage;;
        \? ) echo "Unknown option: -${OPTARG}" >&2; exit 1;;
        :  ) echo "Missing option argument for -${OPTARG}" >&2; exit 1;;
        *  ) echo "Unimplemented option: -${OPTARG}" >&2; exit 1;;
    esac
done

shift $((OPTIND - 1))

if ! dataset_valid "${1}"; then
    die "'${1}' is not a valid dataset (kaggle | random | toy)"
fi

if ! "${build}" && ! "${run}"; then
    die "Neither -b or -r were used, doing nothing"
    exit 0
fi

dataset="${1}"

main
