#include "indexing.h"
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h> // Include Thrust sort
#include <cuda_runtime.h>
#include <iostream>

// Count the number of newline characters '\n'
int count_rows_fastq(const thrust::device_vector<char> &d_data) {
    int row_count = thrust::count(d_data.begin(), d_data.end(), '\n');
    return row_count;
}

// CUDA kernel to find newline indices using grid-stride loop
__global__ void find_newline_indices(
        const char *d_data,
        int *fields_indices,
        int *counter,
        size_t data_size
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < data_size; i += stride) {
        if (d_data[i] == '\n' && i + 1 < data_size) {
            int write_idx = atomicAdd(counter, 1);
            fields_indices[write_idx] = i + 1;
        }
    }
}

// Function to index newline characters and sort them
void indexing_fields(
        const thrust::device_vector<char> &d_data,
        int row_count,
        thrust::device_vector<int> &fields_indices
) {
    // Allocate memory for indices and counter on the device
    thrust::device_vector<int> d_counter(1, 0); // Counter initialized to zero
    // Allocate maximum possible size; will resize later based on actual count
    fields_indices.resize(row_count);

    // Launch kernel
    const int threads_per_block = 256;
    const int blocks = (d_data.size() + threads_per_block - 1) / threads_per_block;

    find_newline_indices<<<blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(d_data.data()),
            thrust::raw_pointer_cast(fields_indices.data()),
            thrust::raw_pointer_cast(d_counter.data()),
            d_data.size()
    );

    // Synchronize to ensure all kernels complete
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error after kernel execution: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel failed");
    }

    // Copy the actual count from device to host
    int h_counter = 0;
    err = cudaMemcpy(&h_counter, thrust::raw_pointer_cast(d_counter.data()), sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error during cudaMemcpy: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMemcpy failed");
    }

    // Resize fields_indices to the actual number of indices found
    fields_indices.resize(h_counter);

    // Include the starting index of the first line (index 0)
    thrust::device_vector<int> temp_indices(h_counter + 1);
    temp_indices[0] = 0; // First line starts at index 0
    thrust::copy(fields_indices.begin(), fields_indices.end(), temp_indices.begin() + 1);
    fields_indices.swap(temp_indices);
    h_counter += 1;

    // Sort the indices using Thrust
    thrust::sort(fields_indices.begin(), fields_indices.end());

    // Synchronize to ensure sorting is complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error after sorting: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Sorting failed");
    }
}
