#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output large_4_thread.log
./threaded datasets/large-train.arff datasets/large-test.arff 3 4