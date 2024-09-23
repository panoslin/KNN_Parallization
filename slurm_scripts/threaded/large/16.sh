#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output log/threaded_large_16
./threaded datasets/large-train.arff datasets/large-test.arff 3 16