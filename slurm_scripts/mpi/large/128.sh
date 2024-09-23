#!/bin/bash
#SBATCH --ntasks 128
#SBATCH --cpus-per-task 1
#SBATCH --mem 5G
#SBATCH --output log/mpi_large_128.log
mpirun -np $SLURM_NTASKS mpi datasets/large-train.arff datasets/large-test.arff 3