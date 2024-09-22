#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output large_32_thread.log
./threaded datasets/large-train.arff datasets/large-test.arff 3 32