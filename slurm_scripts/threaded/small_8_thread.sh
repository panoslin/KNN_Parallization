#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 0.5G
#SBATCH --output small_8_thread.log
./threaded datasets/small-train.arff datasets/small-test.arff 3 8