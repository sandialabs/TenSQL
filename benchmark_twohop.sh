#!/bin/bash

set -ex

lscpu
lsmem

NUM_RUNS=4

export PYTHONPATH="$PYTHONPATH:$(pwd)"
BASEDIR="/srv/tmp/`whoami`"
RESULTSDIR="results/mxm_$(date +%F_%H%M)"

BENCHMARKS=(
    tensql
    postgres_orm
    sqlite_orm
    pandasql
)

for RUN in `seq $NUM_RUNS` ; do
    for BENCHMARK in "${BENCHMARKS[@]}" ; do 
        sbatch ./single_mxm.sh "$BASEDIR" "$RESULTSDIR" "$BENCHMARK"
    done
done
