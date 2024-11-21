# !bin/bash

# Define a list of integers
n_shot_list=(0 2 4 8)

# Iterate over the list
for num in "${n_shot_list[@]}"; do
    sbatch vqa_coco_solver_vila.sh $num 1
    sbatch vqa_coco_solver_vila.sh $num
    # Add additional processing here as needed
done
# n_random_list=(2 4 6)
# for num in "${n_random_list[@]}"; do
#     sbatch vqa_coco_vary_random_solver.sh 8 $num
#     # Add additional processing here as needed
# done