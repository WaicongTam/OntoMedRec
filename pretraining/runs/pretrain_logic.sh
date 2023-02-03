#!/bin/bash
#SBATCH --job-name=pretraining_TaxoMedRec
#SBATCH --account=ar57
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

# Load CUDA
# You can choose the versions available on M3
module load cuda/11.4
module load cudnn/8.0.5-cuda11

# Running your code with GPU
python src/pretraining/pretrain_logic_taxo_indi_encode.py --embd_mode random --scorer mlp --n_epochs 5