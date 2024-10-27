#!/bin/bash
sizes=("small" "medium" "large")


for size in "${sizes[@]}"; do
    for count in {1..5}; do 
        file="slurm_scripts/cuda/${size}/${count}.sh"
        echo "Submitting: sbatch $file"
        sbatch "$file"
    done
done
