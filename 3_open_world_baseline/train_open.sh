#!/bin/bash
#SBATCH --job-name=train_open
#SBATCH -o /home/timothy.walsh/VF/3_open_world_baseline/%x.out
#SBATCH --partition=beards
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=0-24:00:00

source /share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/bin/activate /share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/envs/tflow
python3 /home/timothy.walsh/VF/3_open_world_baseline/train_open.py
