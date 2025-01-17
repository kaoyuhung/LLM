#!/bin/bash

if [ $# -lt 9 ]; then
  echo "Usage: $0 <nnodes> <nproc> <node_rank> <master_addr:master_port> <mode> <eval_nItrs> <model> <model_path> <model_version>"
  exit 1
fi

nnodes=${1}
nproc=${2}
if [ ! -z "${SLURM_NODEID}" ]; then
  node_rank=${SLURM_NODEID}
else
  node_rank=${3}
fi
master_addr=${4%%:*}
master_port=${4##*:}
mode=${5}
eval_nItrs=${6}
model=${7}
model_path=${8}
model_version=${9}

if [ ! -z "$SLURM_NODEID" ]; then
  output_folder="result/job${SLURM_JOB_ID}"
  output_file="${mode}_${model}_${model_version}_node${SLURM_NODEID}_multinode.txt"
else
  output_folder="result"
  output_file="${mode}_${model}_${model_version}_multinode.txt"
fi
mkdir -p $output_folder
if [ -e "$output_folder/$output_file" ]; then
  rm $output_folder/$output_file
fi

export OMP_NUM_THREADS=4

CMD="torchrun \
    --nnodes=$nnodes \
    --nproc-per-node=$nproc \
    --node-rank=$node_rank \
    --master-addr=$master_addr \
    --master-port=$master_port \
    --max-restarts=3 \
    run_mistral.py --mode $mode --model $model --model_path $model_path --model_version $model_version --eval_nItrs $eval_nItrs"

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
