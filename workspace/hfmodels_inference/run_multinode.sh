#!/bin/bash

if [ $# -lt 8 ]; then
  echo "Usage: $0 <nnodes> <nproc> <node_rank> <master_addr:master_port> <mode> <eval_nItrs> <model> <model_version>"
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
model_version=${8}

if [ ! -z "$SLURM_NODEID" ]; then
  output_folder="result/job${SLURM_JOB_ID}"
  output_file="${mode}_${model}_${model_version}_node${SLURM_NODEID}_multinode.txt"
else
  output_folder="result/${mode}_${model}_multinode"
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
    run.py --mode $mode --model $model --model_version $model_version --eval_nItrs $eval_nItrs"

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
