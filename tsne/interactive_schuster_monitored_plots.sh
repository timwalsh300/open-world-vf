#!/bin/bash
#SBATCH --job-name=interactive_plots
#SBATCH -o /home/timothy.walsh/VF/tsne/%x.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=0-8:00:00

python3 /home/timothy.walsh/VF/tsne/interactive_schuster_monitored_plots.py
