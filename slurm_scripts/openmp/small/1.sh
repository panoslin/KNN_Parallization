#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output log/openmp_small_1_thread.log
./openmp datasets/small-train.arff datasets/small-test.arff 3 1