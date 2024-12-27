#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Usage: $0 <nproc> <eval_nItrs>"
  exit 1
fi

nproc=$1
eval_nItrs=$2
batch_size=$3
output_folder="result"
output_file="measure_mixtral8x7B_single_node.txt"
mkdir -p $output_folder

if [ -e "$output_folder/$output_file" ]; then
  rm $output_folder/$output_file
fi

export OMP_NUM_THREADS=4

for N in 1 2 4 8 16
do
  torchrun --standalone --nnodes=1 --nproc-per-node=$nproc \
    run_mistral.py --model "Mixtral-8x7B-Instruct-v0.1" --model_path "weights/Mixtral-8x7B-Instruct-v0.1" --eval_nItrs $eval_nItrs --batch_size $N \
    >> $output_folder/$output_file
done