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
