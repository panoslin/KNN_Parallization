#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 128
#SBATCH --mem 1G
#SBATCH --output log/threaded_large_128
./threaded datasets/large-train.arff datasets/large-test.arff 3 128