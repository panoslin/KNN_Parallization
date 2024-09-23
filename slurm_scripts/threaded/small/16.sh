#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output log/threaded_small_16
./threaded datasets/small-train.arff datasets/small-test.arff 3 16