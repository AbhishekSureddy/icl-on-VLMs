#!/bin/bash

# Define the arrays
n_shots=(0 1 2 4)
styles=("vqa_style" "mcq_style")

# Outer loop for n_shots
for shots in "${n_shots[@]}"; do
    # Inner loop for styles
    for style in "${styles[@]}"; do
        # Call the other script with the current arguments
        # echo "$style $shots"
        sbatch keypoint_detection_solver.sh $style $shots
    done
done
