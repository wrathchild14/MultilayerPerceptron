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
} mlp;

mlp* create_mlp(int input_size, int hidden_size, int output_size);
void free_mlp(mlp* network);

double activation(double x);
double activation_derivative(double x);

void forward(const mlp* network, const double* input);
void backward(const mlp* network, const double* input, const double* target, double learning_rate);

void train(mlp* network, double** inputs, double** labels, int num_samples, double learning_rate, int epochs);

#endif /* MLP_H */
