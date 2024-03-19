#!/bin/bash
#SBATCH --job-name=tar_monitored_https
#SBATCH -o %x.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0-72:00:00

tar cf /data/timothy.walsh/monitored_https.tar -C /data/timothy.walsh/July2023 monitored_https
