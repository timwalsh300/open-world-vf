#!/bin/bash
#SBATCH --job-name=hayden_monitored_https
#SBATCH -o %x.out
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=0-24:00:00

python3 raw_to_csv.py /data/timothy.walsh/July2023 10 hayden monitored https
