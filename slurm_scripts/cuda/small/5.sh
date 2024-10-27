#!/bin/bash
#SBATCH --tasks-per-node=1
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --output=log/cuda/cuda_small_1_5
#SBATCH --partition=gpu-h100 
#SBATCH --gres=gpu:1
module load cuda/12.3
./cuda datasets/small-train.arff datasets/small-test.arff