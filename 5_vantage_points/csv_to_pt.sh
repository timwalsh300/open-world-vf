#!/bin/bash
#SBATCH --job-name=csv_to_pt
#SBATCH -o /home/timothy.walsh/VF/5_vantage_points/data_splits/%x.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=0-12:00:00

source /usr/share/Modules/init/bash
module load lang/python/3.10.10
python3 /home/timothy.walsh/VF/5_vantage_points/data_splits/csv_to_pt.py
