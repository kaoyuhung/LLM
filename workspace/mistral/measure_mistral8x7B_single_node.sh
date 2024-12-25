#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Usage: $0 <nproc> <eval_nItrs>"
  exit 1
fi

nproc=$1
eval_nItrs=$2
batch_size=$3
output_folder="result"
output_file="measure_mistral8x7B_single_node.txt"
mkdir -p $output_folder

if [ -e "$output_folder/$output_file" ]; then
  rm $output_folder/$output_file
fi

for N in 1 2 4 8 16
do
  torchrun --standalone --nnodes=1 --nproc_per_node=$nproc \
    run_mistral.py --model "Mistral-8x7B-Instruct-v0.1" --model_path "mistral_weights/Mistral-8x7B-Instruct-v0.1" --eval_nItrs $eval_nItrs --batch_size $N \
    >> $output_folder/$output_file
done


# torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=$nproc \
# run_mistral.py --model "Mistral-7B-Instruct-v0.3" --model_path "mistral_weights/Mistral-7B-Instruct-v0.3" --eval_nItrs 1\
# > "measure_mistral7B.txt"