#!/bin/bash
#SBATCH --ntasks 32
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G
#SBATCH --nodes 1
#SBATCH --output log/mpi_large_32.log
mpirun -np $SLURM_NTASKS mpi datasets/large-train.arff datasets/large-test.arff 3