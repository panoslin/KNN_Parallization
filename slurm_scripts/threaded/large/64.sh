#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 64
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output large_64_thread.log
./threaded datasets/large-train.arff datasets/large-test.arff 3 64