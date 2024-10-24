#include <iostream>
#include "./kernel.cuh"

int main()
{
    const int arraySize = 5;

    float input[arraySize] = { 1., 2., 3., 4., 5. };
    float output[arraySize] = { 0 };

    activationFunction(input, output, arraySize);

    std::cout << "Output: " << output[0] << ", " << output[1] << ", " << output[2] << ", " << output[3] << ", " << output[4];
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu