#!/bin/bash

gcc -O2 --openmp src/main.c src/mlp.c -o mlp_out -lm
srun --reservation=fri mlp_out
rm mlp_out