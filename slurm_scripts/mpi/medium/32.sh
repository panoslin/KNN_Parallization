#!/bin/bash
#SBATCH --ntasks 32
#SBATCH --cpus-per-task 1
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output mpi_medium_32.log
mpirun -np $SLURM_NTASKS mpi datasets/medium-train.arff datasets/medium-test.arff 3