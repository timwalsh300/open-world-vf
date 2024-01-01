#!/bin/bash
#SBATCH --job-name=search_open_schuster8_https
#SBATCH -o /home/timothy.walsh/VF/3_open_world_baseline/%x.out
#SBATCH --partition=beards
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --time=0-24:00:00

source /share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/bin/activate /share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/envs/tflow
python3 /home/timothy.walsh/VF/3_open_world_baseline/search_open.py schuster8 https
