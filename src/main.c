#include <stdio.h>
#include <stdlib.h>

#include "mlp.h"

int main(void)
{
	mlp* mlp = create_mlp(4, 10, 3);

	const double input[4] = {0.5, 0.3, -0.2, 0.7};

	double* output = forward(mlp, input);

	printf("output: ");
	for (int i = 0; i < mlp->output_size; i++)
	{
		printf("%.4f ", output[i]);
	}
	printf("\n");

	// free
	free(output);
	free_mlp(mlp);

	return 0;
}
