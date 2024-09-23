#!/bin/bash
#SBATCH --ntasks 2
#SBATCH --cpus-per-task 1
#SBATCH --mem 2G
#SBATCH --nodes 1
#SBATCH --output log/mpi_small_2.log
mpirun -np $SLURM_NTASKS mpi datasets/small-train.arff datasets/small-test.arff 3 