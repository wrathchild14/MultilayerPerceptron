#include "mlp.h"

__device__ double atomicAddDouble(double *address, double val)
{
	auto address_as_ull = (unsigned long long *)address;
	unsigned long long old_val = *address_as_ull, new_val;
	do
	{
		new_val = __double_as_longlong(__longlong_as_double(old_val) + val);
	} while (atomicCAS(address_as_ull, old_val, new_val) != old_val);
	return __longlong_as_double(old_val);
}

__device__ double activation(const double x)
{
	return tanh(x);
}

__device__ double activation_derivative(const double x)
{
	return 1 - pow(tanh(x), 2);
}

__global__ void forward_kernel(const int input_size, const int hidden_size, const int output_size,
							   const double *input, const double *w1, const double *w2,
							   const double *b1, const double *b2, double *hidden, double *output)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < hidden_size)
	{
		double sum = 0.0;
		for (int j = 0; j < input_size; j++)
		{
			sum += input[j] * w1[j * hidden_size + i];
		}
		hidden[i] = activation(sum + b1[i]);
	}

	if (i < output_size)
	{
		double sum = 0.0;
		for (int j = 0; j < hidden_size; j++)
		{
			sum += hidden[j] * w2[j * output_size + i];
		}
		output[i] = activation(sum + b2[i]);
	}
}

__global__ void backward_kernel(const int input_size, const int hidden_size, const int output_size,
								const double *input, const double *target, const double *w1, const double *w2,
								const double *b1, const double *b2, double *hidden, double *output,
								double *w1_gradient, double *w2_gradient, double *b1_gradient, double *b2_gradient)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// output error terms
	if (i < output_size)
	{
		const double output1 = output[i];
		const double output_error = (output1 - target[i]) * activation_derivative(output1);
		for (int j = 0; j < hidden_size; j++)
		{
			atomicAddDouble(&w2_gradient[j * output_size + i], output_error * hidden[j]);
		}
		atomicAddDouble(&b2_gradient[i], output_error);
	}

	// hidden error terms
	if (i < hidden_size)
	{
		double error = 0.0;
		for (int j = 0; j < output_size; j++)
		{
			error += (output[j] - target[j]) * w2[i * output_size + j];
		}
		const double hidden_error = error * activation_derivative(hidden[i]);
		for (int j = 0; j < input_size; j++)
		{
			atomicAddDouble(&w1_gradient[j * hidden_size + i], hidden_error * input[j]);
		}
		atomicAddDouble(&b1_gradient[i], hidden_error);
	}
}

void train(mlp *network, double **inputs, double **labels, int num_samples, double learning_rate,
		   int epochs, int batch_size)
{
	double *d_w1, *d_w2, *d_w1_gradient, *d_w2_gradient;
	double *d_b1, *d_b2, *d_hidden, *d_output;

	cudaMalloc((void **)&d_w1, network->input_size * network->hidden_size * sizeof(double));
	cudaMalloc((void **)&d_w2, network->hidden_size * network->output_size * sizeof(double));
	cudaMalloc((void **)&d_w1_gradient, network->input_size * network->hidden_size * sizeof(double));
	cudaMalloc((void **)&d_w2_gradient, network->hidden_size * network->output_size * sizeof(double));
	cudaMalloc((void **)&d_b1, network->hidden_size * sizeof(double));
	cudaMalloc((void **)&d_b2, network->output_size * sizeof(double));
	cudaMalloc((void **)&d_hidden, network->hidden_size * sizeof(double));
	cudaMalloc((void **)&d_output, network->output_size * sizeof(double));

	// host to gpu
	cudaMemcpy(d_w1, network->w1[0], network->input_size * network->hidden_size * sizeof(double),
			   cudaMemcpyHostToDevice);
	cudaMemcpy(d_w2, network->w2[0], network->hidden_size * network->output_size * sizeof(double),
			   cudaMemcpyHostToDevice);
	cudaMemcpy(d_b1, network->b1, network->hidden_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b2, network->b2, network->output_size * sizeof(double), cudaMemcpyHostToDevice);

	for (int epoch = 0; epoch < epochs; epoch++)
	{
		for (int batch = 0; batch < num_samples; batch += batch_size)
		{
			int end = batch + batch_size;
			if (end > num_samples)
				end = num_samples;

			double *d_w1_gradient, *d_w2_gradient;
			double *d_b1_gradient, *d_b2_gradient;

			cudaMalloc((void **)&d_w1_gradient, network->input_size * network->hidden_size * sizeof(double));
			cudaMalloc((void **)&d_w2_gradient, network->hidden_size * network->output_size * sizeof(double));
			cudaMalloc((void **)&d_b1_gradient, network->hidden_size * sizeof(double));
			cudaMalloc((void **)&d_b2_gradient, network->output_size * sizeof(double));

			// init gradients to zero
			cudaMemset(d_w1_gradient, 0, network->input_size * network->hidden_size * sizeof(double));
			cudaMemset(d_w2_gradient, 0, network->hidden_size * network->output_size * sizeof(double));
			cudaMemset(d_b1_gradient, 0, network->hidden_size * sizeof(double));
			cudaMemset(d_b2_gradient, 0, network->output_size * sizeof(double));

			// weights to gpu
			cudaMemcpy(d_w1_gradient, network->w1, network->input_size * sizeof(double *), cudaMemcpyHostToDevice);
			cudaMemcpy(d_w2_gradient, network->w2, network->hidden_size * sizeof(double *), cudaMemcpyHostToDevice);

			for (int sample = batch; sample < end; sample++)
			{
				const double *input = inputs[sample];
				const double *target = labels[sample];

				forward_kernel<<<network->hidden_size / 256 + 1, 256>>>(network->input_size, network->hidden_size,
																		network->output_size, input, d_w1, d_w2,
																		d_b1, d_b2, d_hidden, d_output);
				cudaDeviceSynchronize();

				backward_kernel<<<network->hidden_size / 256 + 1, 256>>>(network->input_size, network->hidden_size,
																		 network->output_size, input, target, d_w1,
																		 d_w2, d_b1, d_b2, d_hidden, d_output,
																		 d_w1_gradient, d_w2_gradient,
																		 d_b1_gradient, d_b2_gradient);
				cudaDeviceSynchronize();
			}

			const double factor = learning_rate / static_cast<double>(batch_size);
			for (int i = 0; i < network->input_size; i++)
			{
				cudaMemcpy(network->w1[i], d_w1, network->hidden_size * sizeof(double), cudaMemcpyDeviceToHost);
				for (int j = 0; j < network->hidden_size; j++)
				{
					network->w1[i][j] -= factor * d_w1_gradient[i * network->hidden_size + j];
				}
			}
			for (int i = 0; i < network->hidden_size; i++)
			{
				cudaMemcpy(network->w2[i], d_w2, network->output_size * sizeof(double), cudaMemcpyDeviceToHost);
				for (int j = 0; j < network->output_size; j++)
				{
					network->w2[i][j] -= factor * d_w2_gradient[i * network->output_size + j];
				}
			}
			cudaMemcpy(network->b1, d_b1, network->hidden_size * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(network->b2, d_b2, network->output_size * sizeof(double), cudaMemcpyDeviceToHost);

			cudaFree(d_w1_gradient);
			cudaFree(d_w2_gradient);
			cudaFree(d_b1_gradient);
			cudaFree(d_b2_gradient);
		}

		printf("epoch %d/%d, loss %lf\n", epoch + 1, epochs, network->loss);
	}

	cudaMemcpy(network->w1, d_w1, network->input_size * network->hidden_size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(network->w2, d_w2, network->hidden_size * network->output_size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(network->b1, d_b1, network->hidden_size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(network->b2, d_b2, network->output_size * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_w1);
	cudaFree(d_w2);
	cudaFree(d_b1);
	cudaFree(d_b2);
	cudaFree(d_hidden);
	cudaFree(d_output);

	// printf("training done in %f s\n", omp_get_wtime() - epochs_dt);
}

mlp *create_mlp(int input_size, int hidden_size, int output_size)
{
	mlp *network = static_cast<mlp *>(malloc(sizeof(mlp)));
	network->input_size = input_size;
	network->hidden_size = hidden_size;
	network->output_size = output_size;

	network->w1 = static_cast<double **>(malloc(input_size * sizeof(double *)));
	network->w2 = static_cast<double **>(malloc(hidden_size * sizeof(double *)));
	network->b1 = static_cast<double *>(malloc(hidden_size * sizeof(double)));
	network->b2 = static_cast<double *>(malloc(output_size * sizeof(double)));

	network->hidden = static_cast<double *>(malloc(hidden_size * sizeof(double)));
	network->output = static_cast<double *>(malloc(output_size * sizeof(double)));

	for (int i = 0; i < input_size; i++)
	{
		network->w1[i] = static_cast<double *>(malloc(hidden_size * sizeof(double)));
	}

	for (int i = 0; i < hidden_size; i++)
	{
		network->w2[i] = static_cast<double *>(malloc(output_size * sizeof(double)));
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

void free_network(mlp *network)
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
