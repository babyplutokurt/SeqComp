#ifndef RLE_H
#define RLE_H

#include <thrust/device_vector.h>

// Function to perform run-length encoding
void rle_compress_int(
        const thrust::device_vector<int> &input,
        thrust::device_vector<int> &unique_keys,
        thrust::device_vector<int> &run_lengths,
        int &num_unique
);

void rle_compress_char(
        const thrust::device_vector<char> &input,   // Input device vector
        thrust::device_vector<char> &unique_keys,  // Output unique keys
        thrust::device_vector<int> &run_lengths,   // Output run lengths
        int &num_unique                            // Output number of unique keys
);

void rle_decompress_char(
        const thrust::device_vector<char> &unique_keys,
        const thrust::device_vector<int> &run_lengths,
        int num_unique,
        thrust::device_vector<char> &output
);

void rle_decompress_memory(
        const std::vector<char> &compressed,   // single memory block in host
        thrust::device_vector<char> &output    // final decompressed data on device
);

#endif // RLE_H
