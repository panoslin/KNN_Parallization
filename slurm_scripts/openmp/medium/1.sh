#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output openmp_medium_1_thread.log
./openmp datasets/medium-train.arff datasets/medium-test.arff 3 1