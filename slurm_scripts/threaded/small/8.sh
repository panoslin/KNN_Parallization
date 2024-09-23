#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output log/threaded_small_8
./threaded datasets/small-train.arff datasets/small-test.arff 3 8