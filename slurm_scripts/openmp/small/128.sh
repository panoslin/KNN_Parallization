#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 128
#SBATCH --mem 1G
#SBATCH --output openmp_small_128_thread.log
./openmp datasets/small-train.arff datasets/small-test.arff 3 128