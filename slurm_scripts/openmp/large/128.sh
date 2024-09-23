#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 128
#SBATCH --mem 4G
#SBATCH --output log/openmp_large_128_thread.log
./openmp datasets/large-train.arff datasets/large-test.arff 3 128