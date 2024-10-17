#!/bin/bash
#SBATCH --job-name=train_baseline_vimeo_oregon
#SBATCH -o /home/timothy.walsh/VF/5_vantage_points/%x.out
#SBATCH --partition=barton
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-3:00:00

source /usr/share/Modules/init/bash
module load lang/python/3.10.10
module load lib/cuda/11.6
python3 /home/timothy.walsh/VF/5_vantage_points/train_baseline_platforms_regions.py
