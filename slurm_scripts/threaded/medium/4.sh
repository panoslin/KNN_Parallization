#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output log/threaded_medium_4
./threaded datasets/medium-train.arff datasets/medium-test.arff 3 4