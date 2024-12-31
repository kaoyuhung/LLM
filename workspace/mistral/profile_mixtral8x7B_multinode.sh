#!/bin/bash

if [ $# -ne 5 ]; then
  echo "Usage: $0 <nnodes> <nproc> <node_rank> <master_addr> <master_port>"
  exit 1
fi


nnodes=$1
nproc=$2
node_rank=$3
master_addr=$4
master_port=$5

torchrun --nnodes=$nnodes --nproc-per-node=$nproc --node-rank=$node_rank --master-addr=$master_addr --master-port=$master_port --max-restarts 3 \
      run_mistral.py --node_rank $node_rank --model "Mixtral-8x7B-Instruct-v0.1" --model_path "weights/Mixtral-8x7B-Instruct-v0.1" --mode "profile" \