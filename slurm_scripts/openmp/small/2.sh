#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output openmp_small_2_thread.log
./openmp datasets/small-train.arff datasets/small-test.arff 3 2