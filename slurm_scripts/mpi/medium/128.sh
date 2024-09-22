#!/bin/bash
#SBATCH --ntasks 128
#SBATCH --cpus-per-task 1
#SBATCH --mem 1G
#SBATCH --output mpi_medium_128.log
mpirun -np $SLURM_NTASKS mpi datasets/medium-train.arff datasets/medium-test.arff 3