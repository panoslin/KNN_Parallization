#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 128
#SBATCH --mem 1G
#SBATCH --output large_128_thread.log
./threaded datasets/large-train.arff datasets/large-test.arff 3 128