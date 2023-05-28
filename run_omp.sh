#!/bin/bash
#SBATCH --job-name=mlp
#SBATCH --output=mlp.out
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --reservation=fri

gcc --openmp src/main.c src/omp.c -o mlp_out -lm
srun --ntasks=1 mlp_out
rm mlp_out