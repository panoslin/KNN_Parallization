#!/bin/bash
#SBATCH --ntasks 128
#SBATCH --cpus-per-task 1
#SBATCH --mem 2G
#SBATCH --output log/mpi_small_128.log
mpirun -np $SLURM_NTASKS mpi datasets/small-train.arff datasets/small-test.arff 3