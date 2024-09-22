#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 1G
#SBATCH --nodes 1
#SBATCH --output mpi_medium_1.log
mpirun -np $SLURM_NTASKS mpi datasets/large-train.arff datasets/large-test.arff 3