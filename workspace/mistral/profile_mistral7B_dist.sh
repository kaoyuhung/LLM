#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <nproc>"
  exit 1
fi

nproc=$1

nsys profile --trace-fork-before-exec=true --cudabacktrace=true --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --sample=none -f true -o profile-mistral-7B-dist\
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=$nproc run_mistral.py --mode nsys_profile --model "Mistral-7B-Instruct-v0.3" --model_path "mistral_weights/Mistral-7B-Instruct-v0.3" --eval_nItrs 1