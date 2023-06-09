#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#include <cuda_runtime.h>

#include "utils.h"

#include <curand_kernel.h>

// debug printing option
#define DEBUG

enum
{
	INPUT_SIZE = 12,
	HIDDEN_SIZE = 20,
	OUTPUT_SIZE = 8,
	DATA_ROWS = 5000,
	BATCH_SIZE_ENUM = 256,
	EPOCHS_ENUM = 10000,
};

const char* DATA_PATH = "data/random_data.txt";
constexpr double LR = 0.00002; // 2e-5

__global__ void initialize_weights(float* weights, int size, unsigned long long seed)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		curandState_t state;
		curand_init(seed, tid, 0, &state);
		float random_value = 2.0 * curand_uniform(&state) - 1.0; // random value between -1 and 1
		weights[tid] = random_value;
	}
}

__global__ void forward_pass(float* d_input_data, float* d_W1, float* d_b1, float* d_W2, float* d_b2,
	float* d_output_layer_output, float* d_hidden_layer_output, int size, int input_size,
	int hidden_size, int output_size, int batch_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int row = tid / hidden_size;
	int col = tid % hidden_size;
	int input_offset = row * input_size + col;
	int hidden_offset = row * hidden_size + col;

	if (tid < size)
	{
		// hidden layer
		float sum = 0.0f;
		for (int i = 0; i < input_size; i++)
		{
			sum += d_input_data[i * batch_size + input_offset] * d_W1[i * hidden_size + col];
		}
		d_hidden_layer_output[hidden_offset] = tanhf(sum + d_b1[col]);

		// output layer
		sum = 0.0f;
		for (int i = 0; i < hidden_size; i++)
		{
			sum += d_hidden_layer_output[i * batch_size + hidden_offset] * d_W2[i * output_size + col];
		}
		d_output_layer_output[hidden_offset] = sum + d_b2[col];
	}
}

__global__ void backward_pass(float* d_output_layer_output, float* d_output_data, float* d_output_layer_error,
	float* d_hidden_layer_output, float* d_hidden_layer_error, int size, int hidden_size,
	int output_size, int batch_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int row = tid / output_size;
	int col = tid % output_size;
	int output_offset = row * output_size + col;
	int hidden_offset = col * batch_size + row;

	if (tid < size)
	{
		// output layer error
		float diff = d_output_layer_output[output_offset] - d_output_data[output_offset];
		d_output_layer_error[output_offset] = diff;

		// hidden layer error
		float val = 1.0f - d_hidden_layer_output[hidden_offset] * d_hidden_layer_output[hidden_offset];
		d_hidden_layer_error[hidden_offset] = val * (d_output_layer_error[output_offset] *
			d_hidden_layer_output[hidden_offset]);
	}
}

__global__ void update_weights(float* d_W1, float* d_b1, float* d_W2, float* d_b2, float* d_input_data,
	float* d_hidden_layer_error, float* d_hidden_layer_output, float* d_output_layer_error,
	float eta, int size, int input_size, int hidden_size, int output_size, int batch_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int row = tid / output_size;
	int col = tid % output_size;

	if (tid < size)
	{
		int input_offset = row * input_size + col;
		int hidden_offset = col * batch_size + row;

		d_W1[input_offset] -= eta * d_hidden_layer_error[hidden_offset] * d_input_data[input_offset];

		if (col == 0)
			d_b1[row] -= eta * d_hidden_layer_error[hidden_offset];

		d_W2[hidden_offset] -= eta * d_output_layer_error[row * output_size + col] *
			d_hidden_layer_output[hidden_offset];

		if (col == 0)
			d_b2[row] -= eta * d_output_layer_error[hidden_offset];
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

	printf("creating network with data_rows:%d batch_size:%d epochs:%d learning_rate:%f\n", DATA_ROWS, BATCH_SIZE,
	       EPOCHS, LR);

	float* d_input_data;
	float* d_output_data;
	float* d_output_layer_output;
	float* d_hidden_layer_output;
	float* d_output_layer_error;
	float* d_hidden_layer_error;

	cudaMalloc(reinterpret_cast<void**>(&d_input_data), DATA_ROWS * INPUT_SIZE * BATCH_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_output_data), DATA_ROWS * OUTPUT_SIZE * BATCH_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_output_layer_output), DATA_ROWS * OUTPUT_SIZE * BATCH_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_hidden_layer_output), DATA_ROWS * HIDDEN_SIZE * BATCH_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_output_layer_error), DATA_ROWS * OUTPUT_SIZE * BATCH_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_hidden_layer_error), DATA_ROWS * HIDDEN_SIZE * BATCH_SIZE * sizeof(float));

	// Copy input/output data for each batch
	for (int i = 0; i < BATCH_SIZE; i++)
	{
		cudaMemcpy(d_input_data + i * DATA_ROWS * INPUT_SIZE, input_data, DATA_ROWS * INPUT_SIZE * sizeof(float),
			cudaMemcpyHostToDevice);
		cudaMemcpy(d_output_data + i * DATA_ROWS * OUTPUT_SIZE, output_data, DATA_ROWS * OUTPUT_SIZE * sizeof(float),
			cudaMemcpyHostToDevice);
	}

	float* d_W1;
	float* d_b1;
	float* d_W2;
	float* d_b2;
	cudaMalloc(reinterpret_cast<void**>(&d_W1), INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_b1), HIDDEN_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_W2), HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&d_b2), OUTPUT_SIZE * sizeof(float));

	cudaMemcpy(d_input_data, input_data, DATA_ROWS * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_output_data, output_data, DATA_ROWS * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	unsigned long long seed = 1234;
	int num_weights = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE;
	dim3 block_size(BATCH_SIZE);
	dim3 grid_size((num_weights + block_size.x - 1) / block_size.x);

	initialize_weights << <grid_size, block_size >> >(d_W1, INPUT_SIZE * HIDDEN_SIZE, seed);
	initialize_weights << <grid_size, block_size >> >(d_b1, HIDDEN_SIZE, seed);
	initialize_weights << <grid_size, block_size >> >(d_W2, HIDDEN_SIZE * OUTPUT_SIZE, seed);
	initialize_weights << <grid_size, block_size >> >(d_b2, OUTPUT_SIZE, seed);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, nullptr);
	for (int epoch = 0; epoch < EPOCHS; epoch++)
	{
		for (int i = 0; i < DATA_ROWS; i += BATCH_SIZE)
		{
			forward_pass << <grid_size, block_size >> > (d_input_data + i * INPUT_SIZE, d_W1, d_b1, d_W2, d_b2,
				d_output_layer_output + i * OUTPUT_SIZE,
				d_hidden_layer_output + i * HIDDEN_SIZE, DATA_ROWS,
				INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);

			backward_pass << <grid_size, block_size >> > (d_output_layer_output + i * OUTPUT_SIZE,
				d_output_data + i * OUTPUT_SIZE,
				d_output_layer_error + i * OUTPUT_SIZE,
				d_hidden_layer_output + i * HIDDEN_SIZE,
				d_hidden_layer_error + i * HIDDEN_SIZE, DATA_ROWS,
				HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);

			update_weights << <grid_size, block_size >> > (d_W1, d_b1, d_W2, d_b2, d_input_data + i * INPUT_SIZE,
				d_hidden_layer_error + i * HIDDEN_SIZE,
				d_hidden_layer_output + i * HIDDEN_SIZE,
				d_output_layer_error + i * OUTPUT_SIZE, LR, DATA_ROWS,
				INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);
		}
#ifdef DEBUG
		printf("Epoch %d done\n", epoch + 1);
#endif
	}
	cudaEventRecord(stop, nullptr);
	cudaEventSynchronize(stop);

	float* h_output_layer_output = new float[DATA_ROWS * OUTPUT_SIZE];
	cudaMemcpy(h_output_layer_output, d_output_layer_output, DATA_ROWS * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	float loss = 0.0f;
	for (int i = 0; i < DATA_ROWS * OUTPUT_SIZE; i++)
	{
		float diff = h_output_layer_output[i] - output_data[i];
		loss += diff * diff;
	}
	loss /= (DATA_ROWS * OUTPUT_SIZE);
	printf("loss: %f\n", loss);

#ifdef DEBUG
	// w and b to host
	auto w1 = static_cast<float*>(malloc(sizeof(float) * INPUT_SIZE * HIDDEN_SIZE));
	auto b1 = static_cast<float*>(malloc(sizeof(float) * HIDDEN_SIZE));
	auto w2 = static_cast<float*>(malloc(sizeof(float) * HIDDEN_SIZE * OUTPUT_SIZE));
	auto b2 = static_cast<float*>(malloc(sizeof(float) * OUTPUT_SIZE));
	cudaMemcpy(w1, d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(b1, d_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(w2, d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(b2, d_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

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

	free(w1);
	free(b1);
	free(w2);
	free(b2);
#endif

	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Elapsed Time: %f\n", static_cast<double>(elapsed_time) / 1000.0);

	cudaFree(d_input_data);
	cudaFree(d_output_data);
	cudaFree(d_W1);
	cudaFree(d_b1);
	cudaFree(d_W2);
	cudaFree(d_b2);
	cudaFree(d_output_layer_output);
	cudaFree(d_hidden_layer_output);
	cudaFree(d_output_layer_error);
	cudaFree(d_hidden_layer_error);

	delete[] input_data;
	delete[] output_data;

	return 0;
}
