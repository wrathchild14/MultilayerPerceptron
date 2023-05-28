#!/bin/sh

module load CUDA/10.1.243-GCC-8.3.0

nvcc -o mlp_out src/cuda/main.cu -lcudart -lm

# Run the executable
srun --reservation=fri -G1 -n1 mlp_out

rm mlp_out
