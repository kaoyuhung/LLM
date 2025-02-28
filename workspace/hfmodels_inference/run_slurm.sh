#!/bin/bash

#Batch Job Paremeters
#SBATCH --account=GOV113121
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1             # one torchrun per node https://stackoverflow.com/a/65897194
#SBATCH --cpus-per-gpu=4
#SBATCH --mail-type=END,BEGIN           # Send the mail when the job starts and finishes.
#SBATCH --mail-user=xxx@xxx.com
#SBATCH --job-name=INFERENCE
#SBATCH --output=result/job%j/log.out

usage="Usage: $0 -m <mode> -M <model> -V <model_version>  \
        [-i <eval_nItrs>] [-b <batch_size>] [-d <dataset>] [-t <max_tokens>]"

while getopts "m:i:b:M:V:d:t:" opt; do
  case $opt in
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
    *)
      echo $usage
      exit 1
      ;;
  esac
done

if [ -z "$mode" ] || [ -z "$model" ] || [ -z "$model_version" ]; then
  echo $usage
  exit 1
fi

# enable NCCL log
# export NCCL_DEBUG=INFO

if [ "$SLURM_JOB_NUM_NODES" -eq 1 ]; then
  SRUN_CMD="./run_single_node.sh -p $SLURM_GPUS_PER_NODE -m $mode -M $model -V $model_version "

else
  # net
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_GPU_DIRECT_RDMA=1 # allows direct memory access between the GPU and the network interface (e.g., Mellanox InfiniBand cards)
  nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
  nodes_array=($nodes)
  head_node=${nodes_array[0]}
  MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
  MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
  SRUN_CMD="./run_multinode.sh -n $SLURM_JOB_NUM_NODES -p $SLURM_GPUS_PER_NODE -a $MASTER_ADDR:$MASTER_PORT -m $mode -M $model -V $model_version " 
fi
if [ ! -z "$eval_nItrs" ]; then
  SRUN_CMD+="-i $eval_nItrs "
fi
if [ ! -z "$batch_size" ]; then
  SRUN_CMD+="-b $batch_size "
fi
if [ ! -z "$dataset" ]; then
  SRUN_CMD+="-d $dataset "
fi
if [ ! -z "$max_tokens" ]; then
  SRUN_CMD+="-t $max_tokens "
fi

# https://discuss.pytorch.org/t/distributed-training-on-slurm-cluster/150417/8
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job ID: $SLURM_JOB_ID"
echo "Job Owner:  $SLURM_JOB_USER"
echo "Nodelist = " $(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "Number of nodes = " $SLURM_JOB_NUM_NODES
echo "Ntasks per node = "  $SLURM_NTASKS_PER_NODE
echo "Gpus per node = " $SLURM_GPUS_PER_NODE
echo "Cpus per GPU = " $SLURM_CPUS_PER_GPU
echo "srun" $SRUN_CMD
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
# Ref: https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp

srun $SRUN_CMD
