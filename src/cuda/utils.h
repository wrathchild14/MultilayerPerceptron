#pragma once
#include <cstdlib>
#include <ctime>

inline void initialize_randomlly(float* weights, float* biases, int input_size, int output_size)
{
	srand(time(nullptr));

	for (int i = 0; i < input_size * output_size; ++i)
	{
		weights[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Random value between -1 and 1
	}

	for (int i = 0; i < output_size; ++i)
	{
		biases[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Random value between -1 and 1
	}
}

inline bool load_data(const char* data_path, const int input_cols, const int output_cols, const int rows,
                      float*& input_data, float*& output_data)
{
	const auto data = static_cast<double**>(malloc(rows * sizeof(double*)));
	for (int i = 0; i < rows; i++)
	{
		data[i] = static_cast<double*>(malloc((input_cols + output_cols) * sizeof(double)));
	}

	input_data = static_cast<float*>(malloc(input_cols * rows * sizeof(float)));
	output_data = static_cast<float*>(malloc(output_cols * rows * sizeof(float)));

	FILE* file = fopen(data_path, "r");

	if (file == nullptr)
	{
		printf("ERROR: failed to open the file.\n");
		return false;
	}

	// read data from the file
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < input_cols + output_cols; j++)
		{
			if (fscanf(file, "%lf,", &data[i][j]) != 1)
			{
				printf("ERROR: failed to read data from the file.\n");
				fclose(file);
				return false;
			}
		}
	}

	fclose(file);

	// fill input and output data
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < input_cols; j++)
		{
			input_data[i * input_cols + j] = static_cast<float>(data[i][j]);
		}

		for (int j = 0; j < output_cols; j++)
		{
			output_data[i * output_cols + j] = static_cast<float>(data[i][input_cols + j]);
		}
	}
	return true;
}
