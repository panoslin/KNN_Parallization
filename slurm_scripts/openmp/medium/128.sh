#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 128
#SBATCH --mem 1G
#SBATCH --output openmp_medium_128_thread.log
./openmp datasets/medium-train.arff datasets/medium-test.arff 3 128