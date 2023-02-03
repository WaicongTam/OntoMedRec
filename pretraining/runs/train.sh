#!/bin/bash
#SBATCH --job-name=training_TaxoMedRec
#SBATCH --account=ar57
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G

# Load CUDA
# You can choose the versions available on M3
module load cuda/11.4
module load cudnn/8.0.5-cuda11

# Running your code with GPU
CUDA_VISIBLE_DEVICES=0 python -u src/train_all.py