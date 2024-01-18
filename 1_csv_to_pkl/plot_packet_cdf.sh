#!/bin/bash
#SBATCH --job-name=plot_packet_cdf
#SBATCH -o /home/timothy.walsh/VF/1_csv_to_pkl/%x.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=0-24:00:00

python3 /home/timothy.walsh/VF/1_csv_to_pkl/plot_packet_cdf.py
