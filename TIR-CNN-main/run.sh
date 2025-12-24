#!/bin/bash

#SBATCH --job-name=himawari-train
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=normal
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --nodelist=gpu01
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00

mkdir -p slurm_logs

echo "Job started on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "-------------------------"
nvidia-smi # 直接运行nvidia-smi来确认节点上是否有GPU
echo "-------------------------"

# --- 加载环境 ---
source "/public/data/anaconda3/etc/profile.d/conda.sh"
conda activate torch

# --- 运行任务 ---
echo "Starting training..."
python3 src/train.py

# --- 任务结束 ---
echo "Job finished at $(date)"
