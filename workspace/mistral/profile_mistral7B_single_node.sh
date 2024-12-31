#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <nproc>"
  exit 1
fi

nproc=$1

output_folder="result"
output_file="profile-mistral7B-single-node"

mkdir -p $output_folder
if [ -e "$output_folder/$output_file" ]; then
  rm $output_folder/$output_file
fi

export OMP_NUM_THREADS=4

nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-memory-usage=true --sample=none -f true -o $output_folder/$output_file \
    torchrun --standalone --nnodes=1 --nproc-per-node=$nproc run_mistral.py --mode nsys_profile --model "Mistral-7B-Instruct-v0.3" --model_path "weights/Mistral-7B-Instruct-v0.3" --eval_nItrs 1