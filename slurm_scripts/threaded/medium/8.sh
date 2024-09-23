#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output log/threaded_medium_8
./threaded datasets/medium-train.arff datasets/medium-test.arff 3 8