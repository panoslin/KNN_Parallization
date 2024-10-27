#!/bin/bash
#SBATCH --tasks-per-node=1
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --output=log/cuda_large_1_2
#SBATCH --partition=gpu-h100 
#SBATCH --gres=gpu:1
./cuda datasets/large-train.arff datasets/large-test.arff