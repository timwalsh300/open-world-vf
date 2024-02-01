#!/bin/bash
#SBATCH --job-name=evaluation_open_trials
#SBATCH -o /home/timothy.walsh/VF/3_open_world_baseline/%x.out
#SBATCH --partition=barton
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=5-0:00:00

source /share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/bin/activate /share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/envs/tflow

echo "Trial 1..."
python3 /home/timothy.walsh/VF/3_open_world_baseline/csv_to_pkl_open.py
python3 /home/timothy.walsh/VF/3_open_world_baseline/train_open.py
python3 /home/timothy.walsh/VF/3_open_world_baseline/evaluation_open.py

echo "Trial 2..."
python3 /home/timothy.walsh/VF/3_open_world_baseline/csv_to_pkl_open.py
python3 /home/timothy.walsh/VF/3_open_world_baseline/train_open.py
python3 /home/timothy.walsh/VF/3_open_world_baseline/evaluation_open.py

echo "Trial 3..."
python3 /home/timothy.walsh/VF/3_open_world_baseline/csv_to_pkl_open.py
python3 /home/timothy.walsh/VF/3_open_world_baseline/train_open.py
python3 /home/timothy.walsh/VF/3_open_world_baseline/evaluation_open.py

echo "Trial 4..."
python3 /home/timothy.walsh/VF/3_open_world_baseline/csv_to_pkl_open.py
python3 /home/timothy.walsh/VF/3_open_world_baseline/train_open.py
python3 /home/timothy.walsh/VF/3_open_world_baseline/evaluation_open.py

echo "Trial 5..."
python3 /home/timothy.walsh/VF/3_open_world_baseline/csv_to_pkl_open.py
python3 /home/timothy.walsh/VF/3_open_world_baseline/train_open.py
python3 /home/timothy.walsh/VF/3_open_world_baseline/evaluation_open.py
