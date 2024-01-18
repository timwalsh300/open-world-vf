#!/bin/bash
#SBATCH --job-name=search_sirinam_vf_https_youtube
#SBATCH -o /home/timothy.walsh/VF/2_closed_world/%x.out
#SBATCH --partition=barton
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx6000:4
#SBATCH --time=0-24:00:00

source /share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/bin/activate /share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/envs/tflow
python3 /home/timothy.walsh/VF/2_closed_world/search.py sirinam_vf https youtube
