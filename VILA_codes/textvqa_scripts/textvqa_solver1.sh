#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40GB  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00  # Job time limit
#SBATCH --constraint a100
#SBATCH -o logs/slurm-%j.out  # %j = job ID


module load conda/latest
cd /home/asureddy_umass_edu/cs682/VILA_codes/llava/eval

conda activate vila

python vila_e2e_vqa_textvqa.py --n_shots=$1 --model_name_or_path="Efficient-Large-Model/VILA1.5-3b"
python vila_e2e_vqa_textvqa.py --n_shots=$1 --model_name_or_path="Efficient-Large-Model/VILA1.5-13b"
