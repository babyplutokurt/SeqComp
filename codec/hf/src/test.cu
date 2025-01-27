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


#include "cusz/type.h"
#include "hfclass.hh"


#define CHECK_CUDA(call)                                             \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// -----------------------------------------------------------------------------
// Shared-memory histogram kernel
// -----------------------------------------------------------------------------
static const int BLOCK_SIZE = 256;

__global__ void histogram256(const unsigned char* data,
                             unsigned int* freq,
                             int dataSize)
{
  // Shared memory for one partial histogram per block
  __shared__ unsigned int sharedHist[256];

  // Global thread index
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Initialize the shared histogram bins to zero
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    sharedHist[i] = 0;
  }
  __syncthreads();

  // Each thread processes elements with a stride
  for (int i = tid; i < dataSize; i += stride) {
    unsigned char val = data[i];
    // Atomically increment the bin in shared memory
    atomicAdd(&sharedHist[val], 1);
  }
  __syncthreads();

  // Write partial histograms to global memory
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    atomicAdd(&freq[i], sharedHist[i]);
  }
}

// -----------------------------------------------------------------------------
// Simple kernel to increment each frequency element by 1
// -----------------------------------------------------------------------------
__global__ void add_one_kernel(unsigned int* freq)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < 256) {
    freq[i] += 1;
  }
}

// -----------------------------------------------------------------------------
// Launch function for histogram256 kernel
// -----------------------------------------------------------------------------
void launch_histogram256(const unsigned char* d_data,
                         unsigned int* d_freq,
                         int dataSize,
                         cudaStream_t stream)
{
  int gridSize = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
  gridSize = std::min(gridSize, 1024);

  histogram256<<<gridSize, BLOCK_SIZE, 0, stream>>>(d_data, d_freq, dataSize);
  CHECK_CUDA(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// Read a file into pinned host memory
// -----------------------------------------------------------------------------
void read_file_to_pinned_buffer(const char* filepath, uint8_t*& h_data, size_t& file_size) {
  std::ifstream infile(filepath, std::ios::binary | std::ios::ate);
  if (!infile) {
    std::cerr << "Error: cannot open file: " << filepath << std::endl;
    exit(EXIT_FAILURE);
  }
  file_size = (size_t)infile.tellg();
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

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <sample_fraction>\n";
    return EXIT_FAILURE;
  }
  const char* input_file = argv[1];
  float sample_fraction = atof(argv[2]);

  // =============================================================
  // Step 1: Read file from disk into pinned host memory
  // =============================================================
  uint8_t *h_data = nullptr;
  size_t INLEN = 0;
  read_file_to_pinned_buffer(input_file, h_data, INLEN);

  // Compute sample data size
  size_t data_sample_size = static_cast<size_t>(INLEN * sample_fraction);
  if (data_sample_size < 1) data_sample_size = 1; // Ensure at least 1 element
  if (data_sample_size > INLEN) data_sample_size = INLEN; // clamp if needed
  std::cout << "Sample fraction: " << sample_fraction
            << ", data sample size: " << data_sample_size << " bytes\n";

  // =============================================================
  // Step 2: Initialize CUDA stream
  // =============================================================
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // =============================================================
  // Step 3: Allocate device memory for the input data
  // =============================================================
  uint8_t *d_data = nullptr;
  CHECK_CUDA(cudaMalloc(&d_data, INLEN * sizeof(uint8_t)));

  // Copy file data from host to device
  CHECK_CUDA(cudaMemcpyAsync(d_data, h_data, INLEN, cudaMemcpyHostToDevice, stream));

  // =============================================================
  // Step 4: Allocate frequency array on GPU
  // =============================================================
  unsigned int *d_freq = nullptr;
  CHECK_CUDA(cudaMalloc(&d_freq, 256 * sizeof(unsigned int)));

  // Initialize freq array on device to 0
  CHECK_CUDA(cudaMemsetAsync(d_freq, 0, 256 * sizeof(unsigned int), stream));

  // =============================================================
  // Step 5: Compute frequency using shared-memory histogram (on sample)
  // =============================================================
  auto start_freq = std::chrono::high_resolution_clock::now();
  launch_histogram256(d_data, d_freq, data_sample_size, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  auto end_freq = std::chrono::high_resolution_clock::now();
  auto duration_freq = std::chrono::duration_cast<std::chrono::milliseconds>(end_freq - start_freq).count();
  std::cout << "Frequency computation time (sample): " << duration_freq << " ms\n";

  // =============================================================
  // Step 5.5: Add 1 to each frequency, so none are zero
  // =============================================================
  {
    add_one_kernel<<<1, 256, 0, stream>>>(d_freq);
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  // =============================================================
  // Step 6: Instantiate HuffmanCodec
  // =============================================================
  size_t inlen = INLEN;
  int bklen = 256;    // up to 256 unique symbols
  int pardeg = 4096;  // example concurrency
  bool debug = false;
  std::cout << "Building Huffman Book...\n";
  phf::HuffmanCodec<u1, true> codec(inlen, bklen, pardeg, debug);

  // =============================================================
  // Step 7: Build the Huffman Book
  // =============================================================
  auto start_build = std::chrono::high_resolution_clock::now();
  codec.buildbook(reinterpret_cast<u4*>(d_freq), reinterpret_cast<phf_stream_t>(stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  auto end_build = std::chrono::high_resolution_clock::now();
  auto duration_build = std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count();
  std::cout << "Build Huffman book time: " << duration_build << " ms\n";

  codec.print_huffman_codes();

  // =============================================================
  // Step 8: Compress (encode)
  // =============================================================
  PHF_BYTE *d_compressed = nullptr;
  size_t compressed_len = 0;

  auto start_encode = std::chrono::high_resolution_clock::now();
  codec.encode(
      reinterpret_cast<u1 *>(d_data), // input GPU pointer
      INLEN,                          // input size (full data)
      &d_compressed,                  // output GPU pointer (allocated by encode)
      &compressed_len,                // output size
      stream
  );
  CHECK_CUDA(cudaStreamSynchronize(stream));
  auto end_encode = std::chrono::high_resolution_clock::now();
  auto duration_encode = std::chrono::duration_cast<std::chrono::milliseconds>(end_encode - start_encode).count();
  std::cout << "Encoding (compress) time: " << duration_encode << " ms\n";
  std::cout << "Compressed size: " << compressed_len << " bytes\n";

  // =============================================================
  // Step 9: Decompress (decode)
  // =============================================================
  uint8_t *d_decompressed = nullptr;
  CHECK_CUDA(cudaMalloc(&d_decompressed, INLEN));

  auto start_decode = std::chrono::high_resolution_clock::now();
  codec.decode(d_compressed, d_decompressed, stream, true);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  auto end_decode = std::chrono::high_resolution_clock::now();
  auto duration_decode = std::chrono::duration_cast<std::chrono::milliseconds>(end_decode - start_decode).count();
  std::cout << "Decoding (decompress) time: " << duration_decode << " ms\n";

  // =============================================================
  // Step 10: Verify correctness
  // =============================================================
  uint8_t *h_decompressed = (uint8_t*)malloc(INLEN);
  CHECK_CUDA(cudaMemcpyAsync(h_decompressed, d_decompressed, INLEN, cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  size_t mismatch_count = 0;
  for (size_t i = 0; i < INLEN; ++i) {
    if (h_decompressed[i] != h_data[i]) {
      mismatch_count++;
    }
  }
  if (mismatch_count == 0) {
    std::cout << "SUCCESS: Decompressed data matches the original file.\n";
  } else {
    std::cout << "FAIL: " << mismatch_count << " mismatches found.\n";
  }

  // =============================================================
  // Cleanup
  // =============================================================
  free(h_decompressed);
  CHECK_CUDA(cudaFreeHost(h_data));
  CHECK_CUDA(cudaFree(d_data));
  CHECK_CUDA(cudaFree(d_freq));
  CHECK_CUDA(cudaFree(d_decompressed));
  CHECK_CUDA(cudaStreamDestroy(stream));

  return EXIT_SUCCESS;
}
