#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output log/openmp_large_8_thread.log
./openmp datasets/large-train.arff datasets/large-test.arff 3 8