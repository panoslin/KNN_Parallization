#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output log/openmp_medium_2_thread.log
./threaded datasets/medium-train.arff datasets/medium-test.arff 3 2