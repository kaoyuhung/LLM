#!/bin/bash

usage="Usage: $0 -n <nnodes> -p <nproc> -r <node_rank> -a <master_addr:master_port> -m <mode> -M <model> -V <model_version>  \
        [-i <eval_nItrs>] [-b <batch_size>] [-d <dataset>] [-t <max_tokens>] [-c <use_cache>]"

if [ ! -z "${SLURM_NODEID}" ]; then
  node_rank=${SLURM_NODEID}
fi

while getopts "n:p:r:a:m:i:b:M:V:d:t:c:" opt; do
  case $opt in
    n)
      nnodes=$OPTARG
      ;;
    p)
      nproc=$OPTARG
      ;;
    r)
      node_rank=$OPTARG
      ;;  
    a)
      master_addr=${OPTARG%%:*}
      master_port=${OPTARG##*:}
      ;;
    m)
      mode=$OPTARG
      ;;
    i)
      eval_nItrs=$OPTARG
      ;;
    b)
      batch_size=$OPTARG
      ;;
    M)
      model=$OPTARG
      ;;
    V)
      model_version=$OPTARG
      ;;
    d)
      dataset=$OPTARG
      ;;
    t)
      max_tokens=$OPTARG
      ;;
    c)
      use_cache=$OPTARG
      ;;
    *)
      echo $usage
      exit 1
      ;;
  esac
done

if [ -z "$nnodes" ] || [ -z "$nproc" ] || [ -z "$node_rank" ] || [ -z "$master_addr" ] || [ -z "$master_port" ]  || [ -z "$mode" ] || [ -z "$model" ] || [ -z "$model_version" ]; then
  echo $usage
  exit 1
fi

if [ ! -z "$SLURM_NODEID" ]; then
  output_folder="result/job${SLURM_JOB_ID}"
  output_file="${mode}_${model}_${model_version}_node${SLURM_NODEID}_multinode.txt"
else
  output_folder="result/${mode}_${model}_multinode"
  output_file="${mode}_${model}_${model_version}_multinode.txt"
  if [ $mode = "eval" ]; then
    if [ -z "$dataset" ]; then
        output_folder+="/mmlu"
    else
        output_folder+="/$dataset"
    fi
  fi
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
    run.py --mode $mode --model $model --model_version $model_version "
if [ ! -z "$eval_nItrs" ]; then
  CMD+="--eval_nItrs $eval_nItrs "
fi
if [ ! -z "$batch_size" ]; then
  CMD+="--batch_size $batch_size "
fi
if [ ! -z "$dataset" ]; then
  CMD+="--dataset $dataset "
fi
if [ ! -z "$max_tokens" ]; then
  CMD+="--max_tokens $max_tokens "
fi
if [ ! -z "$use_cache" ]; then
  CMD+="--use_cache $use_cache "
fi


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
    $CMD

else
  $CMD >> $output_folder/$output_file
  
fi
