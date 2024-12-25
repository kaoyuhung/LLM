#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <nproc>"
  exit 1
fi

nproc=$1

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=$nproc run_mistral.py --model Mixtral-8x7B-Instruct-v0.1 --model_path "mistral_weights/Mixtral-8x7B-Instruct-v0.1" --node-id 0 --batch-size 1 --max_tokens 128
