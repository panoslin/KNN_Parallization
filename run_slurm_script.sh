#!/bin/bash

binaries=("threaded" "openmp" "mpi")
sizes=("small" "medium" "large")
threads=(1 2 4 8 16 32 64 128)

for binary in "${binaries[@]}"; do
    mkdir -p "slurm_scripts/${binary}"  # Use -p to create parent directories if they don't exist
    for size in "${sizes[@]}"; do
        mkdir -p "slurm_scripts/${binary}/${size}"  # Use -p to create the directory if it doesn't exist
        for thread in "${threads[@]}"; do
            file="slurm_scripts/${binary}/${size}/${thread}.sh"
            echo "Submitting: sbatch $file"
            sbatch "$file"
            # exit
        done
    done
done