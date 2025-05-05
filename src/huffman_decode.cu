// huffman_decode.cu

#include "huffman_decode.h"

// CUDA Kernel Implementations

__global__ void histogram256(const unsigned char* data,
                             unsigned int* freq,
                             int dataSize)
{
    // Shared memory for one partial histogram per block
    __shared__ unsigned int sharedHist[256];

    // Initialize shared histogram
    int tid = threadIdx.x;
    for (int i = tid; i < 256; i += blockDim.x) {
        sharedHist[i] = 0;
    }
    __syncthreads();

    // Calculate global thread index and stride
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Populate shared histogram
    for (int i = global_tid; i < dataSize; i += stride) {
        unsigned char val = data[i];
        atomicAdd(&sharedHist[val], 1);
    }
    __syncthreads();

    // Merge shared histogram into global frequency array
    for (int i = tid; i < 256; i += blockDim.x) {
        atomicAdd(&freq[i], sharedHist[i]);
    }
}

__global__ void add_one_kernel(unsigned int* freq)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < 256) {
        freq[i] += 1;
    }
}

// Host Function Implementations

void launch_histogram256(const unsigned char* d_data,
                         unsigned int* d_freq,
                         int dataSize,
                         cudaStream_t stream)
{
    // Calculate grid size based on data size and block size
    int gridSize = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gridSize = std::min(gridSize, 1024); // Limit grid size to 1024 blocks

    // Launch the histogram256 kernel using the constant block size
    histogram256<<<gridSize, BLOCK_SIZE, 0, stream>>>(d_data, d_freq, dataSize);
    CHECK_CUDA(cudaGetLastError());
}

void read_file_to_pinned_buffer(const char* filepath, uint8_t*& h_data, size_t& file_size) {
    std::ifstream infile(filepath, std::ios::binary | std::ios::ate);
    if (!infile) {
        std::cerr << "Error: cannot open file: " << filepath << std::endl;
        exit(EXIT_FAILURE);
    }
    file_size = static_cast<size_t>(infile.tellg());
    infile.seekg(0, std::ios::beg);

    CHECK_CUDA(cudaMallocHost(&h_data, file_size * sizeof(uint8_t)));

    infile.read(reinterpret_cast<char*>(h_data), file_size);
    infile.close();
    if (!infile) {
        std::cerr << "Error: failed to read entire file: " << filepath << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "File '" << filepath << "' size: " << file_size << " bytes\n";
}

// Optionally, include the main function here or keep it separate.
// Refer to the previous response for the main function implementation.

