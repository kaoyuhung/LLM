#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Usage: $0 <eval_nItrs> <batchsize>"
  exit 1
fi

eval_nItrs=$1
batchsize=$2

# torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=$nproc \
# run_mistral.py --model "Mistral-7B-Instruct-v0.3" --model_path "mistral_weights/Mistral-7B-Instruct-v0.3" --eval_nItrs 1\
# > "measure_mistral7B.txt"

output_folder="result"
output_file="measure_mistral7B_single_gpu.txt"
mkdir -p $output_folder

python3 run_mistral.py --model "Mistral-7B-Instruct-v0.3" --model_path "mistral_weights/Mistral-7B-Instruct-v0.3" --eval_nItrs $eval_nItrs  --batch_size $batchsize\
  > $output_folder/$output_file