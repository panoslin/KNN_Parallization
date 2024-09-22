#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 64
#SBATCH --mem 1G
#SBATCH --output openmp_small_64_thread.log
./openmp datasets/small-train.arff datasets/small-test.arff 3 64