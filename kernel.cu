# include "cuda_runtime.h"
# include "device_launch_parameters.h"
# include <stdio.h>
# include <cmath>

__global__ void sigmoidActivation(float* z_matrix, float* activation_matrix) {
    int index = threadIdx.x;
    activation_matrix[index] = 1.0 / (1.0 + exp(-z_matrix[index]));
}

void activationFunction(float* host_z_values, float* host_activations, int arraySize) {
    //float host_z_values[arraySize] = { 1., 2., 3., 4., 5. };
    //float host_activations[arraySize] = { 0 };

    const size_t bytes_z_values = arraySize * sizeof(float);
    const size_t bytes_activations = arraySize * sizeof(float);

    float* device_z_values, * device_activations;

    cudaMalloc(&device_z_values, bytes_z_values);
    cudaMalloc(&device_activations, bytes_activations);
    cudaMemcpy(device_z_values, host_z_values, bytes_z_values, cudaMemcpyHostToDevice);

    sigmoidActivation <<< 1, arraySize >>>(device_z_values, device_activations);

    cudaMemcpy(host_activations, device_activations, bytes_z_values, cudaMemcpyDeviceToHost);
}