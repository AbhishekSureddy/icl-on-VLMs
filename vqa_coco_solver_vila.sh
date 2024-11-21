#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40GB  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00  # Job time limit
#SBATCH --constraint a100
#SBATCH -o slurm-%j.out  # %j = job ID


module load conda/latest
cd /home/asureddy_umass_edu/cs682/VILA/llava/eval

conda activate vila
if [ "$#" -ne 2 ]; then
    python vila_e2e_vqa_coco.py --n_shots $1
else
    python vila_e2e_vqa_coco.py --n_shots $1 --use_random
fi
