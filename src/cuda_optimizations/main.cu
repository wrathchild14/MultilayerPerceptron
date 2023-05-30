#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#include <cuda_runtime.h>

#include "utils.h"

// debug printing option
#define DEBUG

enum
{
	INPUT_SIZE = 12,
	HIDDEN_SIZE = 20,
	OUTPUT_SIZE = 8,
	DATA_ROWS = 5000,
	BATCH_SIZE_ENUM = 256,
	EPOCHS_ENUM = 1000,
};

const char* DATA_PATH = "data/random_data.txt";
constexpr double LR = 0.00002; // 2e-5

__global__ void calculate_loss(float* d_output_layer_output, float* d_output_data, float* d_loss, int size,
                               int output_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		float diff = d_output_layer_output[tid % output_size] - d_output_data[tid];
		d_loss[tid] = 0.5f * diff * diff;
	}
}

__global__ void activation_and_derivative(float* d_input, int size, bool derivative = false)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		float val = tanhf(d_input[tid]);
		if (derivative)
			d_input[tid] = 1.0f - val * val;
		else
			d_input[tid] = val;
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

__global__ void element_wise_operation(float* d_A, float* d_B, float* d_C, int size, int operation)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		if (operation == 0) // Addition
			d_C[tid] = d_A[tid] + d_B[tid];
		else if (operation == 1) // Subtraction
			d_C[tid] = d_A[tid] - d_B[tid];
		else if (operation == 2) // Multiplication
			d_C[tid] = d_A[tid] * d_B[tid];
	}
}

__global__ void update_weights_biases(float* d_W1, float* d_b1, float* d_W2, float* d_b2,
                                      float* d_W1g, float* d_b1g, float* d_W2g, float* d_b2g,
                                      float eta, int input_size, int hidden_size, int output_size, int batch_size)
{
	// sync threads not needed
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < input_size * hidden_size)
	{
		int row = idx / hidden_size;
		int col = idx % hidden_size;
		int weight_idx = row * hidden_size + col;
		d_W1[weight_idx] -= eta * d_W1g[weight_idx] / static_cast<float>(batch_size);
	}

	if (idx < hidden_size)
	{
		d_b1[idx] -= eta * d_b1g[idx];
	}

	if (idx < hidden_size * output_size)
	{
		int row = idx / output_size;
		int col = idx % output_size;
		int weight_idx = row * output_size + col;
		d_W2[weight_idx] -= eta * d_W2g[weight_idx] / static_cast<float>(batch_size);
	}

	if (idx < output_size)
	{
		d_b2[idx] -= eta * d_b2g[idx];
	}
}

int main(int argc, char* argv[])
{
	int BATCH_SIZE = BATCH_SIZE_ENUM;
	int EPOCHS = EPOCHS_ENUM;

	parse_arguments(argc, argv, BATCH_SIZE, EPOCHS);

	const int INPUT_COLS = INPUT_SIZE;
	const int OUTPUT_COLS = OUTPUT_SIZE;

	// handle input/output data
	float* input_data;
	float* output_data;

	if (!load_data(DATA_PATH, INPUT_COLS, OUTPUT_COLS, DATA_ROWS, input_data, output_data)) return 1;

	// int num_samples = sizeof(inputData) / (sizeof(float) * INPUT_SIZE);
	printf("creating network with data_rows:%d batch_size:%d epochs:%d learning_rate:%f\n", DATA_ROWS, BATCH_SIZE,
	       EPOCHS, LR);

	float* d_input_data;
	float* d_output_data;
	cudaMalloc(reinterpret_cast<void**>(&d_input_data), DATA_ROWS * INPUT_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_output_data), DATA_ROWS * OUTPUT_SIZE * sizeof(float));

	cudaMemcpy(d_input_data, input_data, DATA_ROWS * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_output_data, output_data, DATA_ROWS * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	float* d_W1;
	float* d_b1;
	float* d_W2;
	float* d_b2;
	cudaMalloc(reinterpret_cast<void**>(&d_W1), INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_b1), HIDDEN_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_W2), HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_b2), OUTPUT_SIZE * sizeof(float));

	// init weights and biases
	auto w1 = static_cast<float*>(malloc(sizeof(float) * INPUT_SIZE * HIDDEN_SIZE));
	auto b1 = static_cast<float*>(malloc(sizeof(float) * HIDDEN_SIZE));
	auto w2 = static_cast<float*>(malloc(sizeof(float) * HIDDEN_SIZE * OUTPUT_SIZE));
	auto b2 = static_cast<float*>(malloc(sizeof(float) * OUTPUT_SIZE));
	initialize_randomlly(w1, b1, INPUT_SIZE, HIDDEN_SIZE);
	initialize_randomlly(w2, b2, HIDDEN_SIZE, OUTPUT_SIZE);

	// copy w and b to device
	cudaMemcpy(d_W1, w1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b1, b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_W2, w2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b2, b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	// block and grid dim for kernels
	dim3 block_size(BATCH_SIZE);
	dim3 grid_size((DATA_ROWS + block_size.x - 1) / block_size.x);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, nullptr);

	float total_loss;
	const auto loss = static_cast<float*>(malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

	for (int epoch = 0; epoch < EPOCHS; epoch++)
	{
		total_loss = 0.0f;
#pragma omp parallel for reduction(+:total_loss)
		for (int i = 0; i < DATA_ROWS; i += BATCH_SIZE)
		{
			// Forward pass ------
			// device memory for hidden layer
			float* d_hidden_layer_output;
			cudaMalloc(reinterpret_cast<void**>(&d_hidden_layer_output), BATCH_SIZE * HIDDEN_SIZE * sizeof(float));

			// calculate hidden layer
			matrix_multiply << <grid_size, block_size >> >(d_input_data + i * INPUT_SIZE, d_W1, d_hidden_layer_output,
			                                               BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
			element_wise_operation << <grid_size, block_size >> >(d_hidden_layer_output, d_b1, d_hidden_layer_output,
			                                                      BATCH_SIZE * HIDDEN_SIZE, 1);
			activation_and_derivative << <grid_size, block_size >> >(d_hidden_layer_output, BATCH_SIZE * HIDDEN_SIZE);

			// device memory for output layer
			float* d_output_layer_output;
			cudaMalloc(reinterpret_cast<void**>(&d_output_layer_output), BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

			// calculate output layer
			matrix_multiply << <grid_size, block_size >> >(d_hidden_layer_output, d_W2, d_output_layer_output,
			                                               BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
			element_wise_operation << <grid_size, block_size >> >(d_output_layer_output, d_b2, d_output_layer_output,
			                                                      BATCH_SIZE * OUTPUT_SIZE, 0);
			// output - target
			element_wise_operation << <grid_size, block_size >> >(d_output_layer_output,
			                                                      d_output_data + i * OUTPUT_SIZE,
			                                                      d_output_layer_output, BATCH_SIZE * OUTPUT_SIZE, 1);
			activation_and_derivative << <grid_size, block_size >> >(d_output_layer_output, BATCH_SIZE * OUTPUT_SIZE);

			// loss for ------ the current batch (todo: improve?)
			float* d_loss;
			cudaMalloc(reinterpret_cast<void**>(&d_loss), BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
			calculate_loss << <grid_size, block_size >> >(d_output_layer_output, d_output_data + i * OUTPUT_SIZE,
			                                              d_loss, BATCH_SIZE * OUTPUT_SIZE, OUTPUT_SIZE);

			// to host
			cudaMemcpy(loss, d_loss, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

			// accumulate loss for current batch
			for (int j = 0; j < BATCH_SIZE * OUTPUT_SIZE; j++)
			{
				total_loss += loss[j];
			}

			// Backward pass ------
			// device memory for output layer error
			float* d_output_layer_error;
			cudaMalloc(reinterpret_cast<void**>(&d_output_layer_error), BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

			// calculate output layer error
			element_wise_operation << <grid_size, block_size >> >(d_output_layer_output,
			                                                      d_output_data + i * OUTPUT_SIZE,
			                                                      d_output_layer_error, BATCH_SIZE * OUTPUT_SIZE, 1);
			activation_and_derivative << <grid_size, block_size >> >(d_output_layer_output, BATCH_SIZE * OUTPUT_SIZE,
			                                                         true);
			element_wise_operation << <grid_size, block_size >> >(d_output_layer_error, d_output_layer_output,
			                                                      d_output_layer_error, BATCH_SIZE * OUTPUT_SIZE, 2);

			// device memory for hidden layer error
			float* d_hidden_layer_error;
			cudaMalloc(reinterpret_cast<void**>(&d_hidden_layer_error), BATCH_SIZE * HIDDEN_SIZE * sizeof(float));

			// calculate hidden layer error
			matrix_multiply << <grid_size, block_size >> >(d_output_layer_error, d_W2, d_hidden_layer_error,
			                                               BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
			activation_and_derivative << <grid_size, block_size >> >(d_hidden_layer_output, BATCH_SIZE * HIDDEN_SIZE,
			                                                         true);
			element_wise_operation << <grid_size, block_size >> >(d_hidden_layer_error, d_hidden_layer_output,
			                                                      d_hidden_layer_error, BATCH_SIZE * HIDDEN_SIZE, 2);

			// update
			update_weights_biases << <grid_size, block_size >> >(d_W1, d_b1, d_W2, d_b2,
			                                                     d_input_data + i * INPUT_SIZE, d_hidden_layer_error,
			                                                     d_hidden_layer_output, d_output_layer_error,
			                                                     LR, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);

			cudaFree(d_loss);
			cudaFree(d_hidden_layer_output);
			cudaFree(d_output_layer_output);
			cudaFree(d_output_layer_error);
			cudaFree(d_hidden_layer_error);
		}
#ifdef DEBUG
		printf("Epoch %d: MSE Loss = %f\n", epoch + 1, total_loss / DATA_ROWS);
#endif
	}

	// w and b to host
	cudaMemcpy(w1, d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(b1, d_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(w2, d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(b2, d_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef DEBUG
	printf("updated W1:\n");
	for (int i = 0; i < INPUT_SIZE; i++)
	{
		for (int j = 0; j < HIDDEN_SIZE; j++)
		{
			printf("%f ", w1[i * HIDDEN_SIZE + j]);
		}
		printf("\n");
	}

	printf("updated b1:\n");
	for (int i = 0; i < HIDDEN_SIZE; i++)
	{
		printf("%f ", b1[i]);
	}
	printf("\n");

	printf("updated W2:\n");
	for (int i = 0; i < HIDDEN_SIZE; i++)
	{
		for (int j = 0; j < OUTPUT_SIZE; j++)
		{
			printf("%f ", w2[i * OUTPUT_SIZE + j]);
		}
		printf("\n");
	}

	printf("updated b2:\n");
	for (int i = 0; i < OUTPUT_SIZE; i++)
	{
		printf("%f ", b2[i]);
	}
	printf("\n");

#endif

	cudaEventRecord(stop, nullptr);
	cudaEventSynchronize(stop);

	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Elapsed Time: %f s with loss: %f\n", elapsed_time / 1000, total_loss / DATA_ROWS);
	// printf("Elapsed Time: %.3f ms with loss: %f\n", elapsed_time, total_loss / DATA_ROWS);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free(loss);

	cudaFree(d_input_data);
	cudaFree(d_output_data);
	cudaFree(d_W1);
	cudaFree(d_b1);
	cudaFree(d_W2);
	cudaFree(d_b2);

	return 0;
}
