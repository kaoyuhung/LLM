#!/bin/bash

if [ $# -lt 8 ]; then
  echo "Usage: $0 <nnodes> <nproc> <node_rank> <master_addr> <mode> <eval_nItrs> <model> <model_path> [<model_version>]"
  exit 1
fi

nnodes=$1
nproc=$2
node_rank=$3
master_addr=$4
mode=$5
eval_nItrs=$6
model=$7
model_path=$8

output_folder="result"
if [ ! -z "$9" ]; then
  model_version_arg="--model_version $9"
  output_file="${mode}_${model}_${9}_multinode.txt"
else
  output_file="${mode}_${model}_multinode.txt"
fi

mkdir -p $output_folder
if [ -e "$output_folder/$output_file" ]; then
  rm $output_folder/$output_file
fi

export OMP_NUM_THREADS=4

if [ $mode = "measure" ]; then
  for N in 1 2 4 8 16 32 64 128 256
  do
    torchrun --nnodes=$nnodes --nproc-per-node=$nproc --node-rank=$node_rank --master-addr=$master_addr --master-port=5000 --max-restarts 3 \
        run_mistral.py --mode $mode --node_rank $node_rank --model $model --model_path $model_path $model_version_arg --eval_nItrs $eval_nItrs --batch_size $N \
        >> $output_folder/$output_file
  done
else
  torchrun --nnodes=$nnodes --nproc-per-node=$nproc --node-rank=$node_rank --master-addr=$master_addr --master-port=5000 --max-restarts 3 \
        run_mistral.py --mode $mode --node_rank $node_rank --model $model --model_path $model_path $model_version_arg --eval_nItrs $eval_nItrs --batch_size 1 \
        >> $output_folder/$output_file
fi
