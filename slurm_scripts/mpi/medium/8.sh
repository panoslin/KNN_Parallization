#!/bin/bash
#SBATCH --ntasks 8
#SBATCH --cpus-per-task 1
#SBATCH --mem 3G
#SBATCH --nodes 1
#SBATCH --output log/mpi_medium_8.log
mpirun -np $SLURM_NTASKS mpi datasets/medium-train.arff datasets/medium-test.arff 3