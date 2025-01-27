#include "lorenzo.h"
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

// Function to compute an inclusive scan (prefix sum)
void inclusive_scan(const thrust::device_vector<int> &input, thrust::device_vector<int> &output) {
    // Resize output to match input size
    output.resize(input.size());

    // Perform an inclusive scan (prefix sum) using Thrust
    thrust::inclusive_scan(input.begin(), input.end(), output.begin());
}

// Kernel to calculate the difference between the current and previous elements (Lorenzo 1D transform)
__global__ void lorenzo_1d_kernel(const int *input, int *output, size_t size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Start from index 1 to avoid unnecessary checks
    if (idx >= 1 && idx < size) {
        output[idx] = input[idx] - input[idx - 1];
    }
}

// Function to compute Lorenzo 1D transformation
void lorenzo_1d(const thrust::device_vector<int> &input, thrust::device_vector<int> &output) {
    // Resize output to match input size
    output.resize(input.size());

    if (input.size() == 0) return; // Handle empty input case

    // Handle the first element directly on the host
    output[0] = input[0];

    // Get raw pointers to device data
    const int *d_input = thrust::raw_pointer_cast(input.data());
    int *d_output = thrust::raw_pointer_cast(output.data());

    // Launch kernel for indices starting from 1
    const int threads_per_block = 256;
    const int blocks = (input.size() + threads_per_block - 1) / threads_per_block;

    lorenzo_1d_kernel<<<blocks, threads_per_block>>>(d_input, d_output, input.size());

    // Synchronize and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error during lorenzo_1d kernel execution: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("lorenzo_1d kernel failed");
    }
}
