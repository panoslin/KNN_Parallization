#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 128
#SBATCH --mem 1G
#SBATCH --output medium_128_thread.log
./threaded datasets/medium-train.arff datasets/medium-test.arff 3 128