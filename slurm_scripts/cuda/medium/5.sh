#!/bin/bash
#SBATCH --tasks-per-node=1
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --output=log/cuda/cuda_medium_1_5
#SBATCH --partition=gpu-h100 
#SBATCH --gres=gpu:1
module load cuda/12.3
./cuda datasets/medium-train.arff datasets/medium-test.arff