#!/bin/bash
#SBATCH --job-name=evaluation_smaller
#SBATCH -o /home/timothy.walsh/VF/2_closed_world/%x.out
#SBATCH --partition=beards
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=0-12:00:00

source /share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/bin/activate /share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/envs/tflow
python3 /home/timothy.walsh/VF/2_closed_world/evaluation_smaller.py
