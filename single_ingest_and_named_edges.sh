#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

set -ex

BASEDIR="$1/$$"
SOCKETDIR="$BASEDIR/postgres/sockets"
ESCAPED_SOCKETDIR="$(echo -n "$SOCKETDIR" | perl -pe 's/\//\\\//g')"
RESULTSDIR="$2"

shift
shift

mkdir -p "$RESULTSDIR"
mkdir -p "$BASEDIR"
rm -rf "$BASEDIR/postgres"
initdb -D "$BASEDIR/postgres"
mkdir -p "$SOCKETDIR"
cp ./postgresql.conf "$BASEDIR/postgres/postgresql.conf"
sed -i -e "s/^unix_socket_directories = .*/unix_socket_directories = '$ESCAPED_SOCKETDIR'/" "$BASEDIR/postgres/postgresql.conf"
postgres -D "$BASEDIR/postgres/" &

for DATASET in facebook twitter gplus ; do
    mkdir -p "$RESULTSDIR/$DATASET"
    for BENCHMARK in "$@" ; do
          PROJECT_DIR="$BASEDIR" POSTGRES_SOCKETDIR="$SOCKETDIR" python3 benchmark/extract_named_edges.py "benchmark/data/${DATASET}_combined.txt" "$BENCHMARK"
    done | tee -a "$RESULTSDIR/$DATASET/results_${SLURM_JOBID}_${SLURM_PROCID}.txt"
done

rm -rf "$BASEDIR"

kill -SIGTERM %1

wait
