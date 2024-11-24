#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o ./jobs/%j.out  # %j = job ID
#SBATCH --constraint=[a100]

module load conda/latest
# /modules/apps/cuda/10.1.243/samples/bin/x86_64/linux/release/deviceQuery
if [ ! -d "./jobs" ]; then
    echo "Creating ./jobs directory..."
    mkdir -p ./jobs
fi

conda activate icl

python "run_code_4.py"