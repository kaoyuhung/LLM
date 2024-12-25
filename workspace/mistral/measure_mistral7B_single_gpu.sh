#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <eval_nItrs>"
  exit 1
fi

eval_nItrs=$1
output_folder="result"
output_file="measure_mistral7B_single_gpu.txt"

mkdir -p $output_folder
if [ -e "$output_folder/$output_file" ]; then
  rm $output_folder/$output_file
fi

for N in 1 2 4 8 16 32 64
do
  python3 run_mistral.py --model "Mistral-7B-Instruct-v0.3" --model_path "mistral_weights/Mistral-7B-Instruct-v0.3" --eval_nItrs $eval_nItrs  --batch_size $N \
  >> $output_folder/$output_file
done

