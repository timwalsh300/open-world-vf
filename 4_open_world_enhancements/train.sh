#!/bin/bash
#SBATCH --job-name=train_baseline_opengan_pix_dfnet_tor_lambda_g
#SBATCH -o /home/timothy.walsh/VF/4_open_world_enhancements/%x.out
#SBATCH --partition=barton
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-72:00:00

source /usr/share/Modules/init/bash
module load lang/python/3.10.10
module load lib/cuda/11.6
python3 /home/timothy.walsh/VF/4_open_world_enhancements/train_baseline_opengan_pix_dfnet.py tor 0.0
python3 /home/timothy.walsh/VF/4_open_world_enhancements/train_baseline_opengan_pix_dfnet.py tor 0.1
python3 /home/timothy.walsh/VF/4_open_world_enhancements/train_baseline_opengan_pix_dfnet.py tor 0.2
python3 /home/timothy.walsh/VF/4_open_world_enhancements/train_baseline_opengan_pix_dfnet.py tor 0.3
python3 /home/timothy.walsh/VF/4_open_world_enhancements/train_baseline_opengan_pix_dfnet.py tor 0.4
python3 /home/timothy.walsh/VF/4_open_world_enhancements/train_baseline_opengan_pix_dfnet.py tor 0.5
python3 /home/timothy.walsh/VF/4_open_world_enhancements/train_baseline_opengan_pix_dfnet.py tor 0.6
python3 /home/timothy.walsh/VF/4_open_world_enhancements/train_baseline_opengan_pix_dfnet.py tor 0.7
python3 /home/timothy.walsh/VF/4_open_world_enhancements/train_baseline_opengan_pix_dfnet.py tor 0.8
python3 /home/timothy.walsh/VF/4_open_world_enhancements/train_baseline_opengan_pix_dfnet.py tor 0.9
python3 /home/timothy.walsh/VF/4_open_world_enhancements/train_baseline_opengan_pix_dfnet.py tor 1.0
