#!/bin/bash
#SBATCH --job-name=p7zip_monitored_tor
#SBATCH -o %x.out
#SBATCH --cpus-per-task=64
#SBATCH --mem=32GB
#SBATCH --time=0-72:00:00

source /usr/share/Modules/init/bash
module load app/p7zip/17.05
7za a -mmt=64 /data/timothy.walsh/monitored_tor.tar.7z /data/timothy.walsh/monitored_tor.tar
