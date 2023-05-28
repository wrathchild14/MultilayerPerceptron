#!/bin/bash
#SBATCH --job-name=mlp
#SBATCH --output=mlp.out
#SBATCH --time=00:09:00
#SBATCH --reservation=fri

module load mpi/openmpi-4.1.3
mpicc src/serial/main.c src/mpi/mpi.c -o mlp_out -lm
srun --reservation=fri --mpi=pmix -n2 -N1 mlp_out
rm mlp_out