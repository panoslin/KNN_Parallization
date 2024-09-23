#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 64
#SBATCH --mem 1G
#SBATCH --output log/threaded_small_64
./threaded datasets/small-train.arff datasets/small-test.arff 3 64