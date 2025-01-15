#!/bin/bash

if [ $# -lt 4 ]; then
  echo "Usage: $0 <mode> <eval_nItrs> <model> <model_path> [<output_dir>]"
  exit 1
fi
mode=$1
eval_nItrs=$2
model=$3
model_path=$4
if [ ! -z "$5" ]; then
  output_folder=$5
else
  output_folder="result"
fi
output_file="${mode}_${model}_single_gpu.txt"

mkdir -p $output_folder
if [ -e "$output_folder/$output_file" ]; then
  rm $output_folder/$output_file
fi

CMD="python3 run_mistral.py --mode $mode --model $model --model_path $model_path --eval_nItrs $eval_nItrs"

if [ $mode = "measure" ]; then
  for N in 1 2 4 8 16 32 64 128 256
  do
      $CMD --batch_size $N >> $output_folder/$output_file
  done

elif [ $mode = "nsys_profile" ]; then
    nsys profile --cudabacktrace=true --capture-range=cudaProfilerApi --capture-range-end=stop --sample=none -f true -o $output_folder/$output_file \
      $CMD --max_tokens 5

elif [ $mode = "profile" ]; then
  $CMD

else
  $CMD >> $output_folder/$output_file
fi
