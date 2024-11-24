#!/bin/bash

# Define the arrays
n_shots=(0 2 4 8)

# Outer loop for n_shots
for shots in "${n_shots[@]}"; do
    sbatch textvqa_solver.sh $shots
    sbatch textvqa_solver1.sh $shots
done
