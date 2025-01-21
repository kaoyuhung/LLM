#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: $0 <nnodes> <nsys> [<gpus-per-node> ] [<node-rank>] [master-addr:master-port]"
  exit 1
fi

nnodes=$1
nsys=$2
gpus=$3

if [ "$nnodes" -ge 1 ] && [ -z "$3" ]; then 
  echo "gpus-per-node not specified"
  exit 1
fi

if [ "$nnodes" -gt 1 ]; then
  if [ -z "$4" ]; then
    echo "node-rank not specified"
    exit 1

  fi

  if [ -z "$5" ]; then
    echo "[master-addr:master-port not specified"
    exit 1

  fi
  
  if [ "$4" = "xxx" ]; then
      node_rank=$SLURM_NODEID
  else
      node_rank=$4
  fi
  output_file="test_memcpy_${nnodes}_node${node_rank}"
  master_addr=${5%%:*}
  master_port=${5##*:}

else
  output_file="test_memcpy_$nnodes"

fi

if [ ! -z "$SLURM_JOB_ID" ]; then
  output_folder="result/job${SLURM_JOB_ID}"

else
  output_folder="result/$nnodes-$nsys"

fi

if [ "$nsys" -eq 0 ]; then
  nsys="False"
  output_file+=".txt"

else
  nsys="True"
 
fi

mkdir -p $output_folder
if [ -e "$output_folder/$output_file" ]; then
  rm $output_folder/$output_file
fi

export OMP_NUM_THREADS=4

if [ "$nnodes" -eq 0 ]; then
  CMD="python3 ./test_memcpy.py --nsys $nsys" 

elif [ "$nnodes" -eq 1 ]; then
  export NCCL_DEBUG="INFO"
  export NCCL_IGNORE_DISABLED_P2P=1
  CMD="torchrun --standalone --nnodes=1 --nproc-per-node=$gpus ./test_memcpy.py --nsys $nsys"
  
else
  export NCCL_DEBUG="INFO"
  export NCCL_IGNORE_DISABLED_P2P=1
  CMD="torchrun --nnodes=$nnodes --nproc-per-node=$gpus --node-rank=$node_rank --master-addr=$master_addr --master-port=$master_port ./test_memcpy.py --nsys $nsys"
 
fi

if [ "$nsys" = "True" ]; then
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --sample=none \
  --gpu-metrics-devices=all --cuda-memory-usage=true \
  --python-backtrace=cuda --trace-fork-before-exec=true \
  -f true -o $output_folder/$output_file $CMD
  
else
  $CMD >> $output_folder/$output_file
  
fi