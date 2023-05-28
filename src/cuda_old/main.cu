#include <stdbool.h>

#include "mlp.h"

bool load_training_data(const char* file_path, double** inputs, double** labels, const int num_samples,
	const int input_size, const int output_size)
{
	FILE* file = fopen(file_path, "r");
	if (file == NULL)
	{
		printf("Error opening file: %s\n", file_path);
		return false;
	}

	for (int i = 0; i < num_samples; i++)
	{
		for (int j = 0; j < input_size; j++)
		{
			fscanf(file, "%lf,", &inputs[i][j]);
		}
		for (int j = 0; j < output_size; j++)
		{
			fscanf(file, "%lf,", &labels[i][j]);
		}
	}

	fclose(file);
	return true;
}

int main(void)
{
	const char* file_path = "data/random_data.txt";
	const int input_size = 4;
	const int output_size = 3;
	const int hidden_size = 10;
	const int num_samples = 1000;
	const int batch_size = 64;
	const int epochs = 100;
	const double learning_rate = 0.001;

	double** inputs = static_cast<double**>(malloc(num_samples * sizeof(double*)));
	double** labels = static_cast<double**>(malloc(num_samples * sizeof(double*)));
	for (int i = 0; i < num_samples; i++)
	{
		inputs[i] = static_cast<double*>(malloc(input_size * sizeof(double)));
		labels[i] = static_cast<double*>(malloc(output_size * sizeof(double)));
	}

	if (load_training_data(file_path, inputs, labels, num_samples, input_size, output_size))
	{
		mlp* network = create_mlp(input_size, hidden_size, output_size);
		train(network, inputs, labels, num_samples, learning_rate, epochs, batch_size);
		free_network(network);
	}
	for (int i = 0; i < num_samples; i++)
	{
		free(inputs[i]);
		free(labels[i]);
	}
	free(inputs);
	free(labels);

	return 0;
}
