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

if [ $# -lt 2 ]; then
  echo "Usage: $0 <nccl> <nsys>"
  exit 1
fi

nccl=$1
nsys=$2

if [ "$SLURM_JOB_NUM_NODES" -eq 1 ]  && [ "$nccl" = "0" ]; then
    SRUN_CMD="./run_memcpy_test.sh 0 $nsys"
   
else
    export UCX_NET_DEVICES=mlx5_0:1
    export UCX_IB_GPU_DIRECT_RDMA=1 # allows direct memory access between the GPU and the network interface (e.g., Mellanox InfiniBand cards)
    nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
    SRUN_CMD="./run_memcpy_test.sh $SLURM_JOB_NUM_NODES $nsys $SLURM_GPUS_PER_NODE xxx $MASTER_ADDR:$MASTER_PORT"
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