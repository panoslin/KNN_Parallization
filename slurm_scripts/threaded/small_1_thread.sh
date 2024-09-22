#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 0.5G
#SBATCH --output small_1_thread.log
./threaded datasets/small-train.arff datasets/small-test.arff 3 1