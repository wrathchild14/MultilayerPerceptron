#include "../serial/mlp.h"
#include "/usr/include/openmpi-x86_64/mpi.h"

mlp* create_mlp(const int input_size, const int hidden_size, const int output_size)
{
	mlp* network = malloc(sizeof(mlp));
	network->input_size = input_size;
	network->hidden_size = hidden_size;
	network->output_size = output_size;

	network->w1 = malloc(input_size * sizeof(double*));
	network->w2 = malloc(hidden_size * sizeof(double*));
	network->b1 = malloc(hidden_size * sizeof(double));
	network->b2 = malloc(output_size * sizeof(double));

	network->hidden = malloc(hidden_size * sizeof(double));
	network->output = malloc(output_size * sizeof(double));

	for (int i = 0; i < input_size; i++)
	{
		network->w1[i] = malloc(hidden_size * sizeof(double));
	}

	for (int i = 0; i < hidden_size; i++)
	{
		network->w2[i] = malloc(output_size * sizeof(double));
	}

	for (int i = 0; i < input_size; i++)
	{
		for (int j = 0; j < hidden_size; j++)
		{
			network->w1[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
		}
	}

	for (int i = 0; i < hidden_size; i++)
	{
		for (int j = 0; j < output_size; j++)
		{
			network->w2[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
		}
	}

	for (int i = 0; i < hidden_size; i++)
	{
		network->b1[i] = 0;
	}

	for (int i = 0; i < output_size; i++)
	{
		network->b2[i] = 0;
	}

	return network;
}

void free_network(mlp* network)
{
	for (int i = 0; i < network->input_size; i++)
	{
		free(network->w1[i]);
	}
	free(network->w1);

	for (int i = 0; i < network->hidden_size; i++)
	{
		free(network->w2[i]);
	}
	free(network->w2);

	free(network->b1);
	free(network->b2);
	free(network->hidden);
	free(network->output);

	free(network);
}

double activation(const double x)
{
	return tanh(x);
}

double activation_derivative(const double x)
{
	return 1 - pow(tanh(x), 2);
}

void mpi_forward(mlp* network, const double* input, int rank, int num_process)
{
	// hidden layer
	double chunk_size = network->hidden_size / num_process;
	double start = chunk_size*rank;

	for (int i = start; i < start + chunk_size; i++)
	{
		double sum = 0;
		for (int j = 0; j < network->input_size; j++)
		{
			sum += input[j] * network->w1[j][i];
		}
		network->hidden[i] = activation(sum + network->b1[i]);
	}


	// output layer
	chunk_size = network->output_size / num_process;
	start = chunk_size*rank;

	for (int i = start; i < start + chunk_size; i++)
	{
		double sum = 0;
		for (int j = 0; j < network->hidden_size; j++)
		{
			sum += network->hidden[j] * network->w2[j][i];
		}
		network->output[i] = activation(sum + network->b2[i]);
	}

	network->loss = 0;
	chunk_size = network->output_size / num_process;
	start = chunk_size*rank;

	for (int i = start; i < start + chunk_size; i++)
	{
		const double output = network->output[i];
		const double target = input[i];
		network->loss += pow(output - target, 2);
	}
	network->loss /= network->output_size;
}


void mpi_backward(const mlp* network, const double* input, const double* target, const double learning_rate, int rank, int num_process)
{
	// memory for error terms
	double* output_error = malloc(network->output_size * sizeof(double));
	double* hidden_error = malloc(network->hidden_size * sizeof(double));

	// output error terms
	double chunk_size = network->output_size / num_process;
	double start = chunk_size*rank;
	for (int i = start; i < start + chunk_size; i++)
	{
		const double output = network->output[i];
		output_error[i] = (output - target[i]) * activation_derivative(output);
	}

	// hidden error terms
	chunk_size = network->hidden_size / num_process;
	start = chunk_size*rank;
	for (int i = start; i < start + chunk_size; i++)
	{
		double error = 0;
		for (int j = 0; j < network->output_size; j++)
		{
			error += output_error[j] * network->w2[i][j];
		}
		hidden_error[i] = error * activation_derivative(network->hidden[i]);
	}

	// weights and biases
	chunk_size = network->input_size / num_process;
	start = chunk_size*rank;
	for (int i = start; i < start + chunk_size; i++)
	{
		for (int j = 0; j < network->hidden_size; j++)
		{
			network->w1[i][j] -= learning_rate * hidden_error[j] * input[i];
		}
	}

	chunk_size = network->hidden_size / num_process;
	start = chunk_size*rank;
	for (int i = start; i < start + chunk_size; i++)
	{
		for (int j = 0; j < network->output_size; j++)
		{
			network->w2[i][j] -= learning_rate * output_error[j] * network->hidden[i];
		}
	}

	chunk_size = network->hidden_size / num_process;
	start = chunk_size*rank;
	for (int i = start; i < start + chunk_size; i++)
	{
		network->b1[i] -= learning_rate * hidden_error[i];
	}

	chunk_size = network->output_size / num_process;
	start = chunk_size*rank;
	for (int i = start; i < start + chunk_size; i++)
	{
		network->b2[i] -= learning_rate * output_error[i];
	}

	free(output_error);
	free(hidden_error);
}

void mpi_train(mlp* network, double** inputs, double** labels, const int num_samples, const double learning_rate,
           const int epochs, const int batch_size, int argc, char *argv[])
{
	int rank; // process id
	int num_process; // total number of processes 
	int source; // sender rank
	int destination; // receiver rank 
	int tag = 0; // message tag 
	char message_buffer[100]; // message buffer 
	MPI_Status status; // message status 
	char node_name[MPI_MAX_PROCESSOR_NAME]; //node name
	int name_len; //true length of node name
    
	// initialize MPI 
	MPI_Init(&argc, &argv);
	// get number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);
	// get process id 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// get node name
	MPI_Get_processor_name(node_name, &name_len );

	const double epochs_dt = MPI_Wtime();
	for (int epoch = 0; epoch < epochs; epoch++)
	{
		double dt = MPI_Wtime();

		for (int batch = 0; batch < num_samples; batch += batch_size)
		{
			int end = batch + batch_size;
			if (end > num_samples)
				end = num_samples;

			for (int sample = batch; sample < end; sample++)
			{
				const double* input = inputs[sample];
				const double* target = labels[sample];

				mpi_forward(network, input, rank, num_process);
				mpi_backward(network, input, target, learning_rate, rank, num_process);
			}
		}

		printf("epoch %d/%d, loss %lf, time %f s\n", epoch + 1, epochs, network->loss, MPI_Wtime() - dt);
	}
	printf("training done in %f s\n", MPI_Wtime() - epochs_dt);
}
