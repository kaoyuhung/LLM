#!/bin/bash

if [ $# -lt 9 ]; then
  echo "Usage: $0 <nnodes> <nproc> <node_rank> <master_addr> <mode> <eval_nItrs> <model> <model_path> <model_version> [<output_dir>]"
  exit 1
fi

nnodes=${1}
nproc=${2}
node_rank=${3}
master_addr=${4}
mode=${5}
eval_nItrs=${6}
model=${7}
model_path=${8}
model_version=${9}

if [ "$node_rank" -eq 0 ]; then
  if [ ! -z "${10}" ]; then
    output_folder=${10}
  else
    output_folder="result"
  fi
  output_file="${mode}_${model}_${model_version}_multinode.txt"

  mkdir -p $output_folder
  if [ -e "$output_folder/$output_file" ]; then
    rm $output_folder/$output_file
  fi
fi

export OMP_NUM_THREADS=4

CMD="torchrun \
    --nnodes=$nnodes \
    --nproc-per-node=$nproc \
    --node-rank=$node_rank \
    --master-addr=$master_addr \
    --master-port=5000 \
    --max-restarts 3 \
    run_mistral.py --mode $mode --node_rank $node_rank --model $model --model_path $model_path $model_version_arg --eval_nItrs $eval_nItrs"

if [ $mode = "measure" ]; then
  for N in 1 2 4 8 16 32 64 128 256
  do
    if [ "$node_rank" -eq 0 ]; then
      $CMD --batch_size $N >> $output_folder/$output_file
    else
      $CMD --batch_size $N
    fi
  done

elif [ $mode = "nsys_profile" ]; then
  nsys profile --cudabacktrace=true --capture-range=cudaProfilerApi --capture-range-end=stop --sample=none -f true -o $output_folder/$output_file \
    $CMD --max_tokens 5

elif [ $mode = "profile" ]; then
  $CMD

else
  if [ "$node_rank" -eq 0 ]; then
    $CMD >> $output_folder/$output_file
  else
    $CMD
  fi
fi
