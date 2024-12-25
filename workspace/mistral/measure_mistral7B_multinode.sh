#!/bin/bash

if [ $# -ne 7 ]; then
  echo "Usage: $0 <nnodes> <nproc> <node_rank> <master_addr> <master_port> <eval_nItrs> <batch_size>"
  exit 1
fi


nnodes=$1
nproc=$2
node_rank=$3
master_addr=$4
master_port=$5
eval_nItrs=$6
batch_size=$7
output_folder="result"
output_file="measure_mistral7B_multinode.txt"

mkdir -p $output_folder
if [ -e "$output_folder/$output_file" ]; then
  rm $output_folder/$output_file
fi

torchrun --nnodes=$nnodes --nproc_per_node=$nproc --node-rank=$node_rank --master-addr=$master_addr --master-port=$master_port --max-restarts 3 \
  run_mistral.py --model "Mistral-7B-Instruct-v0.3" --model_path "mistral_weights/Mistral-7B-Instruct-v0.3" --eval_nItrs $eval_nItrs --batch_size $batch_size \
  >> $output_folder/$output_file

# torchrun --standalone --nnodes=1 --nproc_per_node=$nproc \
#   run_mistral.py --model "Mistral-7B-Instruct-v0.3" --model_path "mistral_weights/Mistral-7B-Instruct-v0.3" --eval_nItrs $eval_nItrs --batch_size $batch_size \
#   > $output_folder/$output_file

