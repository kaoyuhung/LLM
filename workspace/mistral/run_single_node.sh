#!/bin/bash

if [ $# -lt 6 ]; then
  echo "Usage: $0 <nproc> <mode> <eval_nItrs> <model> <model_path> <model_version>"
  exit 1
fi

nproc=${1}
mode=${2}
eval_nItrs=${3}
model=${4}
model_path=${5}
model_version=${6}
  
if [ ! -z "$SLURM_NODEID" ]; then
  output_folder="result/job${SLURM_JOB_ID}"
else
  output_folder="result"
fi
output_file="${mode}_${model}_${model_version}_single_node.txt"

mkdir -p $output_folder
if [ -e "$output_folder/$output_file" ]; then
  rm $output_folder/$output_file
fi

export OMP_NUM_THREADS=4

CMD="torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$nproc \
    run_mistral.py --mode $mode --model $model --model_path $model_path --model_version $model_version --eval_nItrs $eval_nItrs"

if [ $mode = "measure" ]; then
   for N in 1 2 4 8 16 32 64 128 256
    do
      $CMD --batch_size $N >> $output_folder/$output_file
    done

elif [ $mode = "nsys_profile" ]; then
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --sample=none \
      --cuda-memory-usage=true \
      --python-backtrace=cuda --trace-fork-before-exec=true \
      -f true -o $output_folder/$output_file \
      $CMD --max_tokens 5 --batch_size 128

elif [ $mode = "profile" ]; then
    $CMD

else
    $CMD >> $output_folder/$output_file
fi

