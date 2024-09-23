#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 64
#SBATCH --mem 3G
#SBATCH --nodes 1
#SBATCH --output log/openmp_large_64_thread.log
./openmp datasets/large-train.arff datasets/large-test.arff 3 64