#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH -o /home/timothy.walsh/VF/2_closed_world/%x.out
#SBATCH --partition=barton
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=7-0:00:00

source /share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/bin/activate /share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/envs/tflow
python3 /home/timothy.walsh/VF/2_closed_world/evaluation.py
