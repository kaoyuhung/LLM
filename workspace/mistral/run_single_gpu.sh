#!/bin/bash

if [ $# -lt 4 ]; then
  echo "Usage: $0 <mode> <eval_nItrs> <model> <model_path> [<model_version>]"
  exit 1
fi

mode=$1
eval_nItrs=$2
model=$3
model_path=$4

output_folder="result"
if [ ! -z "$5" ]; then
  model_version_arg="--model_version $5"
  output_file="${mode}_${model}_${5}_single_gpu.txt"
else
  output_file="${mode}_${model}_single_gpu.txt"
fi

mkdir -p $output_folder
if [ -e "$output_folder/$output_file" ]; then
  rm $output_folder/$output_file
fi

if [ $mode = "measure" ]; then
  for N in 1 2 4 8 16 32 64 128 256
  do
    python3 run_mistral.py --mode $mode --model $model --model_path $model_path --eval_nItrs $eval_nItrs --batch_size $N $model_version_arg \
      >> $output_folder/$output_file
  done

elif [ $mode = "nsys_profile" ]; then
    nsys profile --cudabacktrace=true --capture-range=cudaProfilerApi --capture-range-end=stop --sample=none -f true -o $output_folder/$output_file \
      python3 run_mistral.py --mode $mode --model $model --model_path $model_path --eval_nItrs $eval_nItrs --batch_size 1 $model_version_arg --max_tokens 5

elif [ $mode = "profile" ]; then
  python3 run_mistral.py --mode $mode --model $model --model_path $model_path --eval_nItrs $eval_nItrs --batch_size 1 $model_version_arg 
fi

else
  python3 run_mistral.py --mode $mode --model $model --model_path $model_path --eval_nItrs $eval_nItrs --batch_size 1 $model_version_arg \
    >> $output_folder/$output_file
fi
