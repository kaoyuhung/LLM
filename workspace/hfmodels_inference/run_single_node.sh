#!/bin/bash
usage="Usage: $0 -p <nproc> -m <mode> -M <model_path> -V <model_version>  \
        [-i <eval_nItrs>] [-b <batch_size>] [-d <dataset>] [-t <max_tokens>]"

while getopts "p:m:i:b:M:V:d:t:" opt; do
  case $opt in
    p)
      nproc=$OPTARG
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
      model_path=$OPTARG
      model=$(basename "$model_path")
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
    *)
      echo $usage
      exit 1
      ;;
  esac
done

if [ -z "$nproc" ] || [ -z "$mode" ] || [ -z "$model_path" ] || [ -z "$model_version" ]; then
  echo $usage
  exit 1
fi

if [ ! -z "$SLURM_NODEID" ]; then
  output_folder="result/job${SLURM_JOB_ID}"
else
  output_folder="result/${mode}_${model}_single_node"
  if [ $mode = "eval" ]; then
    if [ -z "$dataset" ]; then
        output_folder+="/mmlu"
    else
        output_folder+="/$dataset"
    fi
  fi
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
    run.py --mode $mode -m $model_path --model_version $model_version "
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

