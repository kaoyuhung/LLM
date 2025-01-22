#!/bin/bash

#Batch Job Paremeters
#SBATCH --account=GOV113121
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1             # one torchrun per node https://stackoverflow.com/a/65897194
#SBATCH --mail-type=END,BEGIN           # Send the mail when the job starts and finishes.
#SBATCH --mail-user=xxx@xxx.com
#SBATCH --job-name=INFERENCE
#SBATCH --output=result/job%j/log.out

export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_GPU_DIRECT_RDMA=1 # allows direct memory access between the GPU and the network interface (e.g., Mellanox InfiniBand cards)
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
MASTER_ADDRES=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I)
MASTER_ADDR=$(echo "$MASTER_ADDRES" | grep -o '\b10\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\b' | head -n 1)
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

SRUN_CMD="./run_iperf.sh $MASTER_ADDR $MASTER_PORT"

# https://discuss.pytorch.org/t/distributed-training-on-slurm-cluster/150417/8
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job ID: $SLURM_JOB_ID"
echo "Job Owner:  $SLURM_JOB_USER"
echo "Nodelist = " $(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "MASTER_ADDR = " $MASTER_ADDR
echo "Number of nodes = " $SLURM_JOB_NUM_NODES
echo "Ntasks per node = "  $SLURM_NTASKS_PER_NODE
echo "Gpus per node = " $SLURM_GPUS_PER_NODE
echo "srun" $SRUN_CMD
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
# Ref: https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp

srun $SRUN_CMD

