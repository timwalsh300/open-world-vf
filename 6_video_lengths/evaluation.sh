#!/bin/bash
#SBATCH --job-name=evaluation_open_lengths_tor
#SBATCH -o /home/timothy.walsh/VF/6_video_lengths/%x.out
#SBATCH --partition=monaco
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-72:00:00

source /usr/share/Modules/init/bash
module load lang/python/3.10.10
module load lib/cuda/11.6
python3 /home/timothy.walsh/VF/6_video_lengths/evaluation_open_lengths.py tor 20
python3 /home/timothy.walsh/VF/6_video_lengths/evaluation_open_lengths.py tor 30
python3 /home/timothy.walsh/VF/6_video_lengths/evaluation_open_lengths.py tor 40
python3 /home/timothy.walsh/VF/6_video_lengths/evaluation_open_lengths.py tor 60
python3 /home/timothy.walsh/VF/6_video_lengths/evaluation_open_lengths.py tor 80
python3 /home/timothy.walsh/VF/6_video_lengths/evaluation_open_lengths.py tor 100
python3 /home/timothy.walsh/VF/6_video_lengths/evaluation_open_lengths.py tor 120
python3 /home/timothy.walsh/VF/6_video_lengths/evaluation_open_lengths.py tor 140
python3 /home/timothy.walsh/VF/6_video_lengths/evaluation_open_lengths.py tor 160
python3 /home/timothy.walsh/VF/6_video_lengths/evaluation_open_lengths.py tor 180
python3 /home/timothy.walsh/VF/6_video_lengths/evaluation_open_lengths.py tor 200
python3 /home/timothy.walsh/VF/6_video_lengths/evaluation_open_lengths.py tor 220
