#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct
{
	int input_size;
	int hidden_size;
	int output_size;
	double** w1;
	double** w2;
	double* b1;
	double* b2;
	double learning_rate;
	double* hidden;
	double* output;
	double loss;
} mlp;

mlp* create_mlp(int input_size, int hidden_size, int output_size);
void free_network(mlp* network);

double activation(double x);
double activation_derivative(double x);

void mpi_forward(mlp* network, const double* input, int rank, int num_process);
void mpi_backward(const mlp* network, const double* input, const double* target, double learning_rate, int rank, int num_process);

void mpi_train(mlp* network, double** inputs, double** labels, int num_samples, double learning_rate,
           int epochs, int batch_size,  int argc, char *argv[]);

#endif /* MLP_H */
