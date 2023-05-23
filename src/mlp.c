#include "mlp.h"

mlp* create_mlp(const int input_size, const int hidden_size, const int output_size)
{
	mlp* mlp = malloc(sizeof(*mlp));
	mlp->input_size = input_size;
	mlp->hidden_size = hidden_size;
	mlp->output_size = output_size;

	mlp->w1 = malloc(input_size * sizeof(double*));
	mlp->w2 = malloc(hidden_size * sizeof(double*));
	mlp->b1 = malloc(hidden_size * sizeof(double));
	mlp->b2 = malloc(output_size * sizeof(double));

	for (int i = 0; i < input_size; i++)
	{
		mlp->w1[i] = malloc(hidden_size * sizeof(double));
	}

	for (int i = 0; i < hidden_size; i++)
	{
		mlp->w2[i] = malloc(output_size * sizeof(double));
	}

	// init weights and biases
	for (int i = 0; i < input_size; i++)
	{
		for (int j = 0; j < hidden_size; j++)
		{
			mlp->w1[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
		}
	}

	for (int i = 0; i < hidden_size; i++)
	{
		for (int j = 0; j < output_size; j++)
		{
			mlp->w2[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
		}
	}

	for (int i = 0; i < hidden_size; i++)
	{
		mlp->b1[i] = 0;
	}

	for (int i = 0; i < output_size; i++)
	{
		mlp->b2[i] = 0;
	}

	return mlp;
}

void free_mlp(mlp* mlp)
{
	for (int i = 0; i < mlp->input_size; i++)
	{
		free(mlp->w1[i]);
	}
	free(mlp->w1);

	for (int i = 0; i < mlp->hidden_size; i++)
	{
		free(mlp->w2[i]);
	}
	free(mlp->w2);

	free(mlp->b1);
	free(mlp->b2);

	free(mlp);
}

double activation(const double x)
{
	return tanh(x);
}

double activation_derivative(const double x)
{
	return 1 - pow(tanh(x), 2);
}

double* forward(const mlp* mlp, const double* input)
{
	double* hidden = malloc(mlp->hidden_size * sizeof(double));
	double* output = malloc(mlp->output_size * sizeof(double));

	for (int i = 0; i < mlp->hidden_size; i++)
	{
		double sum = 0;
		for (int j = 0; j < mlp->input_size; j++)
		{
			sum += input[j] * mlp->w1[j][i];
		}
		hidden[i] = activation(sum + mlp->b1[i]);
	}

	for (int i = 0; i < mlp->output_size; i++)
	{
		double sum = 0;
		for (int j = 0; j < mlp->hidden_size; j++)
		{
			sum += hidden[j] * mlp->w2[j][i];
		}
		output[i] = activation(sum + mlp->b2[i]);
	}

	free(hidden);
	return output;
}
