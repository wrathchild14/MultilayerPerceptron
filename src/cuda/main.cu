#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#include <cuda_runtime.h>

#define DEBUG 1

enum
{
	INPUT_SIZE = 4,
	HIDDEN_SIZE = 8,
	OUTPUT_SIZE = 3,
	DATA_ROWS = 1000,
	BATCH_SIZE = 128,
	EPOCHS = 1000,
};

char const* DATA_PATH = "data/random_data.txt";
constexpr float LR = 0.001f;

__global__ void calculate_loss(float* d_output_layer_output, float* d_output_data, float* d_loss, int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		float diff = d_output_layer_output[tid] - d_output_data[tid];
		d_loss[tid] = 0.5f * diff * diff;
	}
}

__global__ void tanh_activation(float* d_input, int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		d_input[tid] = tanhf(d_input[tid]);
	}
}

__global__ void tanh_derivative(float* d_input, int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		float tanh_val = tanhf(d_input[tid]);
		d_input[tid] = 1.0f - tanh_val * tanh_val;
	}
}

// matrix multiplication (dot product)
__global__ void matrix_multiply(float* d_A, float* d_B, float* d_C, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < k)
	{
		float sum = 0.0f;
		for (int i = 0; i < n; i++)
		{
			sum += d_A[row * n + i] * d_B[i * k + col];
		}
		d_C[row * k + col] = sum;
	}
}

__global__ void element_wise_subtract(float* d_A, float* d_B, float* d_C, int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		d_C[tid] = d_A[tid] - d_B[tid];
	}
}

__global__ void element_wise_multiply(float* d_A, float* d_B, float* d_C, int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		d_C[tid] = d_A[tid] * d_B[tid];
	}
}

__global__ void update_weights_biases(float* d_W1, float* d_b1, float* d_W2, float* d_b2,
                                      float* d_W1g, float* d_b1g, float* d_W2g, float* d_b2g,
                                      float eta, int input_size, int hidden_size, int output_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < input_size * hidden_size)
	{
		int row = idx / hidden_size;
		int col = idx % hidden_size;
		d_W1[idx] -= eta * d_W1g[idx];
	}

	if (idx < hidden_size)
	{
		d_b1[idx] -= eta * d_b1g[idx];
	}

	if (idx < hidden_size * output_size)
	{
		int row = idx / output_size;
		int col = idx % output_size;
		d_W2[idx] -= eta * d_W2g[idx];
	}

	if (idx < output_size)
	{
		d_b2[idx] -= eta * d_b2g[idx];
	}
}


int main()
{
	int INPUT_COLS = INPUT_SIZE;
	int OUTPUT_COLS = OUTPUT_SIZE;
	int ROWS = 1000;

	auto data = static_cast<double**>(malloc(ROWS * sizeof(double*)));
	for (int i = 0; i < ROWS; i++)
	{
		data[i] = static_cast<double*>(malloc((INPUT_COLS + OUTPUT_COLS) * sizeof(double)));
	}

	auto input_data = static_cast<float*>(malloc(INPUT_COLS * ROWS * sizeof(float)));
	auto output_data = static_cast<float*>(malloc(OUTPUT_COLS * ROWS * sizeof(float)));

	FILE* file = fopen(DATA_PATH, "r");

	if (file == nullptr)
	{
		printf("ERROR: failed to open the file.\n");
		return 1;
	}

	// read data from the file
	for (int i = 0; i < ROWS; i++)
	{
		for (int j = 0; j < INPUT_COLS + OUTPUT_COLS; j++)
		{
			if (fscanf(file, "%lf,", &data[i][j]) != 1)
			{
				printf("ERROR: failed to read data from the file.\n");
				fclose(file);
				return 1;
			}
		}
	}

	fclose(file);

	// fill input and output data
	for (int i = 0; i < ROWS; i++)
	{
		for (int j = 0; j < INPUT_COLS; j++)
		{
			input_data[i * INPUT_COLS + j] = static_cast<float>(data[i][j]);
		}

		for (int j = 0; j < OUTPUT_COLS; j++)
		{
			output_data[i * OUTPUT_COLS + j] = static_cast<float>(data[i][INPUT_COLS + j]);
		}
	}

	// int num_samples = sizeof(inputData) / (sizeof(float) * INPUT_SIZE);
	int num_samples = ROWS;
	printf("samples: %d\n", num_samples);

	float* d_input_data;
	float* d_output_data;
	cudaMalloc(reinterpret_cast<void**>(&d_input_data), num_samples * INPUT_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_output_data), num_samples * OUTPUT_SIZE * sizeof(float));

	cudaMemcpy(d_input_data, input_data, num_samples * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_output_data, output_data, num_samples * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	float* d_W1;
	float* d_b1;
	float* d_W2;
	float* d_b2;
	cudaMalloc(reinterpret_cast<void**>(&d_W1), INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_b1), HIDDEN_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_W2), HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_b2), OUTPUT_SIZE * sizeof(float));

	// init weights and biases
	float W1[INPUT_SIZE][HIDDEN_SIZE] = {
		{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f},
		{0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f},
		{0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f},
		{0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f}
	};
	float b1[HIDDEN_SIZE] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
	float W2[HIDDEN_SIZE][OUTPUT_SIZE] = {
		{0.1f, 0.2f, 0.3f},
		{0.2f, 0.3f, 0.4f},
		{0.3f, 0.4f, 0.5f},
		{0.4f, 0.5f, 0.6f},
		{0.5f, 0.6f, 0.7f},
		{0.6f, 0.7f, 0.8f},
		{0.7f, 0.8f, 0.9f},
		{0.8f, 0.9f, 1.0f}
	};
	float b2[OUTPUT_SIZE] = {0.1f, 0.2f, 0.3f};

	// copy w and b to device
	cudaMemcpy(d_W1, W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b1, b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_W2, W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b2, b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);


	// block and grid dim for kernels
	dim3 block_size(256);
	dim3 grid_size((num_samples + block_size.x - 1) / block_size.x);

	for (int epoch = 0; epoch < EPOCHS; epoch++)
	{
		float total_loss = 0.0f; // accumulation loss for epoch
		for (int i = 0; i < num_samples; i += BATCH_SIZE)
		{
			// Forward pass ------
			// device memory for hidden layer
			float* d_hidden_layer_output;
			cudaMalloc(reinterpret_cast<void**>(&d_hidden_layer_output), BATCH_SIZE * HIDDEN_SIZE * sizeof(float));

			// calculate hidden layer
			matrix_multiply << <grid_size, block_size >> >(d_input_data + i * INPUT_SIZE, d_W1, d_hidden_layer_output,
			                                               BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
			element_wise_subtract << <grid_size, block_size >> >(d_hidden_layer_output, d_b1, d_hidden_layer_output,
			                                                     BATCH_SIZE * HIDDEN_SIZE);
			tanh_activation << <grid_size, block_size >> >(d_hidden_layer_output, BATCH_SIZE * HIDDEN_SIZE);

			// device memory for output layer
			float* d_output_layer_output;
			cudaMalloc(reinterpret_cast<void**>(&d_output_layer_output), BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

			// calculate output layer
			matrix_multiply << <grid_size, block_size >> >(d_hidden_layer_output, d_W2, d_output_layer_output,
			                                               BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
			element_wise_subtract << <grid_size, block_size >> >(d_output_layer_output, d_b2, d_output_layer_output,
			                                                     BATCH_SIZE * OUTPUT_SIZE);
			tanh_activation << <grid_size, block_size >> >(d_output_layer_output, BATCH_SIZE * OUTPUT_SIZE);


			// Backward pass ------
			// device memory for output layer error
			float* d_output_layer_error;
			cudaMalloc(reinterpret_cast<void**>(&d_output_layer_error), BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

			// calculate output layer error
			element_wise_subtract << <grid_size, block_size >> >(d_output_layer_output, d_output_data + i * OUTPUT_SIZE,
			                                                     d_output_layer_error, BATCH_SIZE * OUTPUT_SIZE);
			tanh_derivative << <grid_size, block_size >> >(d_output_layer_output, BATCH_SIZE * OUTPUT_SIZE);
			element_wise_multiply << <grid_size, block_size >> >(d_output_layer_error, d_output_layer_output,
			                                                     d_output_layer_error, BATCH_SIZE * OUTPUT_SIZE);

			// device memory for hidden layer error
			float* d_hidden_layer_error;
			cudaMalloc(reinterpret_cast<void**>(&d_hidden_layer_error), BATCH_SIZE * HIDDEN_SIZE * sizeof(float));

			// calculate hidden layer error
			matrix_multiply << <grid_size, block_size >> >(d_output_layer_error, d_W2, d_hidden_layer_error,
			                                               BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
			tanh_derivative << <grid_size, block_size >> >(d_hidden_layer_output, BATCH_SIZE * HIDDEN_SIZE);
			element_wise_multiply << <grid_size, block_size >> >(d_hidden_layer_error, d_hidden_layer_output,
			                                                     d_hidden_layer_error, BATCH_SIZE * HIDDEN_SIZE);

			// update
			update_weights_biases << <grid_size, block_size >> >(d_W1, d_b1, d_W2, d_b2,
			                                                     d_input_data + i * INPUT_SIZE, d_hidden_layer_error,
			                                                     d_hidden_layer_output, d_output_layer_error,
			                                                     LR, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

			cudaFree(d_hidden_layer_output);
			cudaFree(d_output_layer_output);
			cudaFree(d_output_layer_error);
			cudaFree(d_hidden_layer_error);
		}
		printf("Epoch %d: Loss = %f\n", epoch + 1, total_loss);
	}

	// w and b to host
	cudaMemcpy(W1, d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(b1, d_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(W2, d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(b2, d_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef DEBUG
	printf("updated W1:\n");
	for (const auto& i : W1)
	{
		for (const float j : i)
		{
			printf("%f ", j);
		}
		printf("\n");
	}

	printf("updated b1:\n");
	for (const float i : b1)
	{
		printf("%f ", i);
	}
	printf("\n");

	printf("updated W2:\n");
	for (const auto& i : W2)
	{
		for (const float j : i)
		{
			printf("%f ", j);
		}
		printf("\n");
	}

	printf("updated b2:\n");
	for (const float i : b2)
	{
		printf("%f ", i);
	}
	printf("\n");
#endif

	cudaFree(d_input_data);
	cudaFree(d_output_data);
	cudaFree(d_W1);
	cudaFree(d_b1);
	cudaFree(d_W2);
	cudaFree(d_b2);

	return 0;
}
