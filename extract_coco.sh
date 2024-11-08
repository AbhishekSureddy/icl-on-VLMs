#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40GB  # Requested Memory
#SBATCH -p cpu  # Partition
#SBATCH -t 02:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

cd /scratch/workspace/asureddy_umass_edu-llm_alignment/dataset
unzip train2017.zip