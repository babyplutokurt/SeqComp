#include <cub/cub.cuh>
#include <thrust/host_vector.h> 
#include <random>
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>

void check_cuda_error(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        printf("CUDA Error: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void test_radix_sort(size_t N) {
    // Step 1: Generate random data on CPU
    std::vector<int> h_input(N);
    std::mt19937 rng(42); // Seeded RNG
    std::uniform_int_distribution<int> dist(0, 1'000'000'000); // Large range: [0, 1,000,000,000]

    for (size_t i = 0; i < N; ++i) {
        h_input[i] = dist(rng);
    }

    // Debug: Print the first 10 unsorted input data
    printf("Input Data (Unsorted):\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_input[i]);
    }
    printf("...\n");

    // Step 2: Allocate and copy data to GPU
    int *d_input, *d_output;
    check_cuda_error(cudaMalloc(&d_input, N * sizeof(int)), "Allocating d_input");
    check_cuda_error(cudaMalloc(&d_output, N * sizeof(int)), "Allocating d_output");
    check_cuda_error(cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice), "Copying data to GPU");

    // Step 3: CUB temporary storage
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Determine temporary storage size
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_input, d_output, N
    );

    // Allocate temporary storage
    check_cuda_error(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Allocating d_temp_storage");

    // Step 4: Measure radix sort time using CUDA events
    cudaEvent_t start, stop;
    check_cuda_error(cudaEventCreate(&start), "Creating start event");
    check_cuda_error(cudaEventCreate(&stop), "Creating stop event");

    // Start timing
    check_cuda_error(cudaEventRecord(start, 0), "Recording start event");
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_input, d_output, N
    );
    check_cuda_error(cudaEventRecord(stop, 0), "Recording stop event");
    check_cuda_error(cudaEventSynchronize(stop), "Synchronizing stop event");

    float elapsed_ms = 0;
    check_cuda_error(cudaEventElapsedTime(&elapsed_ms, start, stop), "Calculating elapsed time");

    printf("Radix sort for %zu elements took %.2f ms\n", N, elapsed_ms);

    // Debug: Print the first 10 sorted data
    thrust::host_vector<int> h_output(N);
    check_cuda_error(cudaMemcpy(h_output.data(), d_output, N * sizeof(int), cudaMemcpyDeviceToHost), "Copying sorted data to CPU");

    printf("Sorted Data:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("...\n");

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp_storage);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    size_t N = 1'000'000'000; // 1 billion elements
    test_radix_sort(N);
    return 0;
}

