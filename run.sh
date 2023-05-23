#!/bin/bash
#SBATCH --job-name=mlp
#SBATCH --output=mlp.out
#SBATCH --time=00:01:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --reservation=fri

gcc src/main.c src/mlp.c -o mlp_out -lm
srun --ntasks=1 mlp_out
rm mlp_out