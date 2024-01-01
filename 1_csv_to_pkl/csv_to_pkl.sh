#!/bin/bash
#SBATCH --job-name=csv_to_pkl
#SBATCH -o /home/timothy.walsh/VF/1_csv_to_pkl/%x.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=0-24:00:00

python3 /home/timothy.walsh/VF/1_csv_to_pkl/csv_to_pkl.py
