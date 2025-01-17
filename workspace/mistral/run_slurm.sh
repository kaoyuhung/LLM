#!/bin/bash

#Batch Job Paremeters
#SBATCH --account=GOV113121
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1             # one torchrun per node https://stackoverflow.com/a/65897194
#SBATCH --cpus-per-gpu=4
#SBATCH --mail-type=END,BEGIN           # Send the mail when the job starts and finishes.
#SBATCH --mail-user=r13922052@csie.ntu.edu.tw
#SBATCH --job-name=INFERENCE
#SBATCH --output=result/job%j/log.out

if [ $# -lt 4 ]; then
  echo "Usage: $0 <mode> <eval_nItrs> <model> <model_path> [<model_version>]"
  exit 1
fi

mode=$1
eval_nItrs=$2
model=$3
model_path=$4
if [ "$SLURM_JOB_NUM_NODES" -gt 1 ] || [ "$SLURM_GPUS_PER_NODE" -gt 1 ]; then
  if [ -z "$5" ]; then
    echo "model_version not specified"
    exit 1
  else
    model_version=$5
  fi
fi
output_folder="result/job$SLURM_JOB_ID"

# enable NCCL log
# export NCCL_DEBUG=INFO

if [ "$SLURM_JOB_NUM_NODES" -eq 1 ]; then
  if [ "$SLURM_GPUS_PER_NODE" -eq 1 ]; then
      SRUN_CMD="./run_single_gpu.sh $mode $eval_nItrs $model $model_path $output_folder"
  else  
      SRUN_CMD="./run_single_node.sh $SLURM_GPUS_PER_NODE $mode $eval_nItrs $model $model_path $model_version $output_folder"
  fi

else
  # net
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_GPU_DIRECT_RDMA=1 # allows direct memory access between the GPU and the network interface (e.g., Mellanox InfiniBand cards)
  nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
  nodes_array=($nodes)
  head_node=${nodes_array[0]}
  MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
  MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
  SRUN_CMD="./run_multinode.sh $SLURM_JOB_NUM_NODES $SLURM_GPUS_PER_NODE xxx $MASTER_ADDR:$MASTER_PORT $mode $eval_nItrs $model $model_path $model_version $output_folder" 
fi

# https://discuss.pytorch.org/t/distributed-training-on-slurm-cluster/150417/8
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job ID: $SLURM_JOB_ID"
echo "Job Owner:  $SLURM_JOB_USER"
echo "Node ID: $SLURM_NODEID"
echo "Nodelist = " $(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "Number of nodes = " $SLURM_JOB_NUM_NODES
echo "Ntasks per node = "  $SLURM_NTASKS_PER_NODE
echo "Gpus per node = " $SLURM_GPUS_PER_NODE
echo "Cpus per GPU = " $SLURM_CPUS_PER_GPU
echo "srun" $SRUN_CMD
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
# Ref: https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp

srun $SRUN_CMD
