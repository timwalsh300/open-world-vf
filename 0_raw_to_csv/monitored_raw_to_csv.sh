#!/bin/bash
#SBATCH --job-name=sirinam_wf_monitored_tor
#SBATCH -o %x.out
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=0-24:00:00

python3 raw_to_csv.py /data/timothy.walsh/July2023 10 sirinam_wf monitored tor
