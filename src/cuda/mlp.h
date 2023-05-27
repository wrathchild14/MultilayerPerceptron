#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <omp.h>

#include <cuda_runtime.h>

typedef struct
{
	int input_size;
	int hidden_size;
	int output_size;
	double **w1;
	double **w2;
	double *b1;
	double *b2;
	double learning_rate;
	double *hidden;
	double *output;
	double loss;
} mlp;

mlp *create_mlp(int input_size, int hidden_size, int output_size);
void free_network(mlp *network);
void train(mlp *network, double **inputs, double **labels, int num_samples, double learning_rate,
           int epochs, int batch_size);

__device__ double atomic_add_double(double *address, double val);
__device__ double activation(const double x);
__device__ double activation_derivative(const double x);

__global__ void forward_kernel(const int input_size, const int hidden_size, const int output_size,
							   const double *input, const double *w1, const double *w2,
							   const double *b1, const double *b2, double *hidden, double *output);
__global__ void backward_kernel(const int input_size, const int hidden_size, const int output_size,
								const double *input, const double *target, const double *w1, const double *w2,
								const double *b1, const double *b2, double *hidden, double *output,
								double *w1_gradient, double *w2_gradient, double *b1_gradient, double *b2_gradient);

#endif /* MLP_H */
