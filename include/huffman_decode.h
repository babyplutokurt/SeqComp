// huffman_decode.h

#ifndef HUFFMAN_CUDA_H
#define HUFFMAN_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <algorithm>

// Include necessary custom headers
#include "cusz/type.h"
#include "hfclass.hh"

// Macro for CUDA error checking
#define CHECK_CUDA(call)                                             \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Define a constant block size
constexpr int BLOCK_SIZE = 256;

// CUDA Kernel Declarations

/**
 * @brief Kernel to compute histogram of unsigned char data.
 *
 * @param data Pointer to input data on device.
 * @param freq Pointer to frequency array on device.
 * @param dataSize Number of elements in input data.
 */
__global__ void histogram256(const unsigned char* data,
                             unsigned int* freq,
                             int dataSize);

/**
 * @brief Kernel to increment each element of the frequency array by one.
 *
 * @param freq Pointer to frequency array on device.
 */
__global__ void add_one_kernel(unsigned int* freq);

// Host Function Declarations

/**
 * @brief Launches the histogram256 kernel.
 *
 * @param d_data Pointer to input data on device.
 * @param d_freq Pointer to frequency array on device.
 * @param dataSize Number of elements in input data.
 * @param stream CUDA stream for kernel execution.
 */
void launch_histogram256(const unsigned char* d_data,
                         unsigned int* d_freq,
                         int dataSize,
                         cudaStream_t stream);

/**
 * @brief Reads a file into pinned host memory.
 *
 * @param filepath Path to the input file.
 * @param h_data Reference to host data pointer.
 * @param file_size Reference to variable storing file size.
 */
void read_file_to_pinned_buffer(const char* filepath, uint8_t*& h_data, size_t& file_size);

// Main Function Declaration (Optional)
// If you plan to have the main function here, otherwise keep it in a separate file.

#endif // HUFFMAN_CUDA_H
