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
} mlp;

mlp* create_mlp(int input_size, int hidden_size, int output_size);

void free_mlp(mlp* mlp);

// currently tanh
double activation(double x);

double activation_derivative(double x);

double* forward(const mlp* mlp, const double* input);

#endif /* MLP_H */
