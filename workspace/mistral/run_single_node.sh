#!/bin/bash

if [ $# -lt 5 ]; then
  echo "Usage: $0 <nproc> <mode> <eval_nItrs> <model> <model_path> [<model_version>]"
  exit 1
fi

nproc=$1
mode=$2
eval_nItrs=$3
model=$4
model_path=$5

output_folder="result"
if [ ! -z "$6" ]; then
  model_version_arg="--model_version $6"
  output_file="${mode}_${model}_${6}_single_node.txt"
else
  output_file="${mode}_${model}_single_node.txt"
fi

mkdir -p $output_folder
if [ -e "$output_folder/$output_file" ]; then
  rm $output_folder/$output_file
fi

export OMP_NUM_THREADS=4

if [ $mode = "measure" ]; then
   for N in 1 2 4 8 16 32 64 128 256
    do
      torchrun --standalone --nnodes=1 --nproc-per-node=$nproc \
        run_mistral.py --mode $mode --model $model --model_path $model_path --eval_nItrs $eval_nItrs --batch_size $N $model_version_arg \
        >> $output_folder/$output_file
    done

elif [ $mode = "nsys_profile" ]; then
    nsys profile --cudabacktrace=true --capture-range=cudaProfilerApi --capture-range-end=stop --sample=none -f true -o $output_folder/$output_file \
      torchrun --standalone --nnodes=1 --nproc-per-node=$nproc \
        run_mistral.py --mode $mode --model $model --model_path $model_path --eval_nItrs $eval_nItrs --batch_size 1 $model_version_arg --max_tokens 5

elif [ $mode = "profile" ]; then
    torchrun --standalone --nnodes=1 --nproc-per-node=$nproc \
      run_mistral.py --mode $mode --model $model --model_path $model_path --eval_nItrs $eval_nItrs --batch_size 1 $model_version_arg

else
    torchrun --standalone --nnodes=1 --nproc-per-node=$nproc \
      run_mistral.py --mode $mode --model $model --model_path $model_path --eval_nItrs $eval_nItrs --batch_size 1 $model_version_arg \
      >> $output_folder/$output_file
fi

