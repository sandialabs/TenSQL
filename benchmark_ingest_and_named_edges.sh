#!/bin/bash

set -ex

lscpu
lsmem
lspci

NUM_RUNS=4

export PYTHONPATH="$PYTHONPATH:$(pwd)"
BASEDIR="/srv/tmp/`whoami`"
RESULTSDIR="results/named_edges_$(date +%F_%H%M)"

BENCHMARKS=(
    tensql
    postgres_orm
    sqlite_orm
    pandasql
)

for RUN in `seq $NUM_RUNS` ; do
    for BENCHMARK in "${BENCHMARKS[@]}" ; do 
        sbatch ./single_named_edges.sh "$BASEDIR" "$RESULTSDIR" "$BENCHMARK"
    done
done
