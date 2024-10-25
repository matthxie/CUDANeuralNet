#include <iostream>
#include "./kernel.cuh"

int main()
{
    const int shapeLength = 4;
	int shape[] = { 4, 8, 8, 1 };
	float *output = nullptr;

    feedForwardNetwork(shape, shapeLength, output);

    std::cout << "Output: " << output[0];
}
