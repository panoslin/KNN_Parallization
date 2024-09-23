#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 128
#SBATCH --mem 1G
#SBATCH --output log/threaded_small_128
./threaded datasets/small-train.arff datasets/small-test.arff 3 128