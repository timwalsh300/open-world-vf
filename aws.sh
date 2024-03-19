#!/bin/bash
#SBATCH --job-name=aws_monitored_tor
#SBATCH -o %x.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-96:00:00

source /usr/share/Modules/init/bash
module load app/aws/2.0.3
aws s3 cp /data/timothy.walsh/monitored_tor.tar.7z s3://open-world-vf/monitored_tor.tar.7z --storage-class DEEP_ARCHIVE
