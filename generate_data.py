import numpy as np

def generate_training_data(file_path, input_size, output_size, num_samples):
    inputs = np.random.rand(num_samples, input_size)
    labels = np.random.rand(num_samples, output_size)

    with open(file_path, 'w') as file:
        for i in range(num_samples):
            input_values = ','.join(str(value) for value in inputs[i])
            label_values = ','.join(str(value) for value in labels[i])
            file.write(f"{input_values},{label_values}\n")

file_path = "random_data.txt"
input_size = 4
output_size = 3
num_samples = 1000

generate_training_data(file_path, input_size, output_size, num_samples)
