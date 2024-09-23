#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 128
#SBATCH --mem 1G
#SBATCH --output log/threaded_medium_128
./threaded datasets/medium-train.arff datasets/medium-test.arff 3 128