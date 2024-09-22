#!/bin/bash
#SBATCH --ntasks 16
#SBATCH --cpus-per-task 1
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output mpi_small_16.log
mpirun -np $SLURM_NTASKS mpi datasets/small-train.arff datasets/small-test.arff 3