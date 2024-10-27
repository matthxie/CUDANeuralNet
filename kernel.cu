#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <iostream>
#include <algorithm>


__global__ void linearLayer(float* weights, float* biases,
	float* z_values, float* activation_values,
	int* shape, int shape_length) {

	int id = threadIdx.x;

	int layer_offset_z = 0;
	int layer_offset_biases = 0;
	int layer_offset_weights = 0;
	int layer_offset_activations_input = 0;
	int layer_offset_activations_current = shape[0] * blockDim.y;

	for (int shape_index = 0; shape_index < shape_length; shape_index++) {
		if (id < shape[shape_index + 1]) {
			int n_layer_inputs = shape[shape_index];
			int layer_size = shape[shape_index + 1];

			for (int neuron_index = 0; neuron_index < n_layer_inputs; neuron_index++) {
				z_values[layer_offset_biases + threadIdx.y * layer_size + id] += weights[layer_offset_weights + (n_layer_inputs)*id + neuron_index] *
					activation_values[layer_offset_activations_input + threadIdx.y * n_layer_inputs + neuron_index];
			}

			z_values[layer_offset_biases + threadIdx.y * layer_size + id] += biases[layer_offset_biases + id];
			activation_values[layer_offset_activations_current + threadIdx.y * layer_size + id] = 1.0 / 
				(1.0 + exp(-z_values[layer_offset_z + threadIdx.y * layer_size + id]));
		}

		layer_offset_z += shape[shape_index + 1] * blockDim.y;
		layer_offset_biases += shape[shape_index + 1];
		layer_offset_weights += shape[shape_index] * shape[shape_index + 1];
		layer_offset_activations_input = layer_offset_activations_current;
		layer_offset_activations_current += shape[shape_index + 1] * blockDim.y;

		__syncthreads();
	}
}

void normalWeightInitialization(float *&weights, float *&biases, float *&host_z, int n_weights, int n_biases, int n_neurons) {
	cudaMalloc((void**)&weights, n_weights * sizeof(float));
	cudaMalloc((void**)&biases, n_biases * sizeof(float));
	cudaMalloc((void**)&host_z, n_biases * sizeof(float));

	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator, time(0));

	curandGenerateNormal(generator, weights, n_weights, 0.0f, 1.0f);
	curandGenerateNormal(generator, biases, n_biases, 0.0f, 1.0f);
	curandGenerateNormal(generator, host_z, n_neurons, 0.0f, 1.0f);

	curandDestroyGenerator(generator);
}

void xavierWeightInitialization(int* shape, float* weights, float* biases, float* host_z, int n_weights, int n_biases) {

}

void feedForwardNetwork(int *shape, int shape_length, float *output, int batch_size) {
	int n_weights = 0;
	int n_biases = 0;
	int n_neurons = 0;

	for (int shape_index = 0; shape_index < shape_length - 1; shape_index++) {
		n_weights += shape[shape_index] * shape[shape_index + 1];
	}

	for (int shape_index = 1; shape_index < shape_length; shape_index++) {
		n_neurons += shape[shape_index];
	}

	n_biases = n_neurons - shape[0];

	float* host_weights = new float[n_weights] {0.0f};
	float* host_biases = new float[n_biases] {0.0f};
	float* host_activations = new float[n_neurons*batch_size] {0.0f};
	float* host_z = new float[n_biases*batch_size] {0.0f};

	//normalWeightInitialization(host_weights, host_biases, host_activations, n_weights, n_biases, n_neurons);

	const size_t bytes_biases = n_biases * sizeof(float);
	const size_t bytes_z = n_biases * sizeof(float);
	const size_t bytes_weights = n_weights * batch_size * sizeof(float);
	const size_t bytes_activations = n_neurons * batch_size * sizeof(float);
	const size_t bytes_shape = sizeof(int) * shape_length;

	float* device_weights, * device_biases, * device_z, * device_activations;
	int* device_shape;
	cudaMalloc(&device_weights, bytes_weights);
	cudaMalloc(&device_biases, bytes_biases);
	cudaMalloc(&device_z, bytes_z);
	cudaMalloc(&device_activations, bytes_activations);
	cudaMalloc(&device_shape, bytes_shape);

	cudaMemcpy(device_weights, host_weights, bytes_weights, cudaMemcpyHostToDevice);
	cudaMemcpy(device_biases, host_biases, bytes_biases, cudaMemcpyHostToDevice);
	cudaMemcpy(device_z, host_z, bytes_z, cudaMemcpyHostToDevice);
	cudaMemcpy(device_activations, host_activations, bytes_activations, cudaMemcpyHostToDevice);
	cudaMemcpy(device_shape, shape, bytes_shape, cudaMemcpyHostToDevice);

	int n_threads = *std::max_element(shape + 1, shape + shape_length);
	dim3 thread_dimensions(n_threads, batch_size);

	linearLayer << <1, thread_dimensions >> > (device_weights, device_biases, device_z, device_activations, device_shape, shape_length);

	cudaMemcpy(host_activations, device_activations, bytes_activations, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_z, device_z, bytes_z, cudaMemcpyDeviceToHost);

	cudaFree(device_weights);
	cudaFree(device_biases);
	cudaFree(device_z);
	cudaFree(device_activations);
	cudaFree(device_shape);

	output = host_activations;

	int activations_offset = shape[0];
	for (int shape_index = 1; shape_index < shape_length; shape_index++)
	{
		std::cout << "Activations " << shape_index << ". hidden layer" << std::endl;

		for (int neuron_nr = 0; neuron_nr < shape[shape_index]; neuron_nr++)
		{
			std::cout << host_activations[neuron_nr + activations_offset] << std::endl;
		}
		activations_offset += shape[shape_index];
	}

	getchar();
}

void backPropagation() {

}