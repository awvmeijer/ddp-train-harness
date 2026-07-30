#!/bin/bash
#SBATCH --job-name=ddp-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1        # one launcher per node; torchrun spawns the workers
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=runs/%x-%j.out

# Multi-node DDP under Slurm. srun starts one torchrun per node; torchrun
# spawns one worker per GPU and wires up RANK / WORLD_SIZE / LOCAL_RANK.
# Partition, account, and GPU type are site-specific: set them at submit time,
#   sbatch --partition=gpu --account=<proj> slurm/submit.sbatch

set -euo pipefail

mkdir -p runs

# Rendezvous on the first node of the allocation.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

# NCCL: fail with a stack trace instead of hanging, and say what it is doing.
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
# If the cluster has InfiniBand, NCCL finds it. To force TCP for debugging:
#   export NCCL_IB_DISABLE=1

srun python -m torch.distributed.run \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node="$SLURM_GPUS_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    src/train.py \
    --backend nccl \
    --epochs 20 \
    --checkpoint-dir "runs/ckpt-$SLURM_JOB_ID" \
    --checkpoint-every 5 \
    --result-file "runs/result-$SLURM_JOB_ID.json"
