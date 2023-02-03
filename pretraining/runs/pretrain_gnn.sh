#!/bin/bash
#SBATCH --job-name=pretraining_TaxoMedRec
#SBATCH --account=ar57
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/ar57_scratch/medication_scratch/miniconda/conda/envs/qen/lib

# Running your code with GPU
python src/pretraining/pretrain_gnns.py --gnn gat --n_epochs 20
python src/pretraining/pretrain_gnns.py --gnn gcn --n_epochs 20