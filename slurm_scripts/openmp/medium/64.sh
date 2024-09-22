#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 64
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output openmp_medium_64_thread.log
./openmp datasets/medium-train.arff datasets/medium-test.arff 3 64