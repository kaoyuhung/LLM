#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <nproc>"
  exit 1
fi

nproc=$1

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=$nproc run_mistral.py --model "Mistral-7B-Instruct-v0.3" --model_path "mistral_weights/Mistral-7B-Instruct-v0.3" --eval_nItrs 1
