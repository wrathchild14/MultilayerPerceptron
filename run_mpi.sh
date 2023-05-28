#!/bin/bash
#SBATCH --job-name=mlp
#SBATCH --output=mlp.out
#SBATCH --time=00:09:00
#SBATCH --reservation=fri

module load OpenMPI/4.1.0-GCC-10.2.0 
mpicc src/mpi/mpi_main.c src/mpi/mpi.c -o mlp_out -lm
srun --reservation=fri --mpi=pmix -n1 -N1 mlp_out
rm mlp_out