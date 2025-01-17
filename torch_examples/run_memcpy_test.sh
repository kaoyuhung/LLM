#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: $0 <nnodes> <gpus-per-node> [<node-rank>] [master-addr:master-port]"
  exit 1
fi

nnodes=$1
gpus=$2
if [ "$nnodes" -gt 1 ]; then
  if [ -z "$3" ]; then
    echo "node-rank not specified"
    exit 1
  else
    if [ "$3" = "xxx" ]; then
        node_rank=$SLURM_NODEID
    else
        node_rank=$3
    fi
  fi
  master_addr=${4%%:*}
  master_port=${4##*:}
fi

export OMP_NUM_THREADS=4

if [ "$nnodes" -eq 1 ]; then
    torchrun --standalone --nnodes=1 --nproc-per-node=$gpus ./test_memcpy.py
else
    torchrun --nnodes=$nnodes --nproc-per-node=$gpus --node-rank=$node_rank --master-addr=$master_addr --master-port=$master_port ./test_memcpy.py
fi