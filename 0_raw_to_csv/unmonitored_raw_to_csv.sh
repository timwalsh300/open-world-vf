#!/bin/bash
#SBATCH --job-name=dschuster8_unmonitored_tor
#SBATCH -o %x.out
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=0-6:00:00

python3 raw_to_csv.py /data/timothy.walsh/July2023 64 dschuster8 unmonitored tor
