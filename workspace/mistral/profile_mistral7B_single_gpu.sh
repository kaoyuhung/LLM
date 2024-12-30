#!/bin/bash

output_folder="result"
output_file="profile-mistral7B-single-gpu"

mkdir -p $output_folder
if [ -e "$output_folder/$output_file" ]; then
  rm $output_folder/$output_file
fi

nsys profile --cudabacktrace=true --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --cuda-memory-usage=true --sample=none -f true -o $output_folder/$output_file \
    python3 run_mistral.py --mode nsys_profile --model "Mistral-7B-Instruct-v0.3" --model_path "weights/Mistral-7B-Instruct-v0.3" --eval_nItrs 1