#include "quality_scores_process.h"
#include "indexing.h"  // Ensure this header is correctly included and paths are set
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <cuda_runtime.h>
#include <iostream>

// Error checking macro
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(-1);                                             \
        }                                                         \
    } while(0)

/******************************************************************************
 * 1) calcQualityScoreLengthsFromFields
 *    quality_length[n] = (fields_indices[4n+4] - 1) - fields_indices[4n+3]
 ******************************************************************************/
__global__ void calcQualityScoreLineLengthsKernel(
        const int *d_fields_indices,
        int *d_quality_length,
        int num_records
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_records) return;

    // Quality score line => 4*idx + 3
    int quality_idx = 4 * idx + 3;

    // Assuming fields_indices has size row_count +1, where row_count = 4 * num_records
    int start = d_fields_indices[quality_idx];
    int end   = d_fields_indices[quality_idx + 1] - 1; // Exclude newline

    // Ensure that end >= start to prevent negative lengths
    int length = (end >= start) ? (end - start) : 0;
    d_quality_length[idx] = length;
}

void calcQualityScoreLengthsFromFields(
        const int *d_fields_indices,
        int *d_quality_length,
        int num_records
)
{
    int blockSize = 256;
    int gridSize = (num_records + blockSize - 1) / blockSize;

    calcQualityScoreLineLengthsKernel<<<gridSize, blockSize>>>(
            d_fields_indices,
            d_quality_length,
            num_records
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/******************************************************************************
 * 2) calcQualityScoreOffsets
 *    exclusive scan of quality_length => offset array
 ******************************************************************************/
void calcQualityScoreOffsets(
        const int *d_quality_length,
        int num_records,
        int *d_offset
)
{
    // Perform an exclusive scan: offset[i] = sum of quality_length[0..i-1], offset[0] = 0
    thrust::device_ptr<const int> d_len_ptr(d_quality_length);
    thrust::device_ptr<int> d_offset_ptr(d_offset);

    thrust::exclusive_scan(d_len_ptr, d_len_ptr + num_records, d_offset_ptr);
}

/******************************************************************************
 * 3) copyQualityScoresKernel
 *    Kernel to copy quality scores into a contiguous device array based on prefix offsets
 ******************************************************************************/
__global__ void copyQualityScoresKernel(
        const char *d_data,
        const int *d_fields_indices,
        const int *d_quality_length,
        const int *d_offset,
        int num_records,
        char *d_compressed
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_records) return;

    // Quality score line => 4*idx + 3
    int quality_idx = 4 * idx + 3;
    int start = d_fields_indices[quality_idx];
    int length = d_quality_length[idx];
    int offset = d_offset[idx]; // Starting position in d_compressed

    // Boundary check: Ensure that offset + length does not exceed allocated memory
    // This assumes that d_offset has been correctly computed and total_scores is accurate
    // For extra safety, we can pass total_scores as an additional parameter or compute it here
    // Since it's not passed, we assume correctness

    for (int i = 0; i < length; ++i) {
        d_compressed[offset + i] = d_data[start + i];
    }
}

/******************************************************************************
 * 4) qualityScoresCompress
 *    Complete compression process for quality scores
 ******************************************************************************/
void qualityScoresCompress(
        const char *d_data,
        const int *d_fields_indices,
        int row_count,
        int num_records,
        QualityScoresCompressMeta &outMeta
)
{
    // 1) Calculate quality score lengths
    int *d_quality_length = nullptr;
    CUDA_CHECK(cudaMalloc(&d_quality_length, num_records * sizeof(int)));
    calcQualityScoreLengthsFromFields(d_fields_indices, d_quality_length, num_records);

    // 2) Calculate prefix sum offsets using exclusive scan
    int *d_offset = nullptr;
    CUDA_CHECK(cudaMalloc(&d_offset, (num_records) * sizeof(int))); // exclusive scan result size = num_records
    calcQualityScoreOffsets(d_quality_length, num_records, d_offset);

    // 3) Retrieve total quality scores from d_offset[num_records -1] + d_quality_length[num_records -1]
    // Since exclusive_scan: d_offset[num_records] is the sum of all quality_length
    // Allocate temporary host variable to hold d_offset[num_records] which is not in d_offset (size is num_records)
    // Thus, we need to perform a scan with num_records +1 elements to get the total
    // Alternatively, compute total_scores on host

    // To get d_offset[num_records], perform an inclusive scan on d_quality_length
    // Or use thrust::reduce
    // Alternatively, launch a kernel to sum the quality_length

    // Simpler approach: Use thrust::reduce to sum d_quality_length
    int total_scores =0;
    CUDA_CHECK(cudaMemcpy(&total_scores, d_quality_length + num_records -1, sizeof(int), cudaMemcpyDeviceToHost));
    int last_length =0;
    CUDA_CHECK(cudaMemcpy(&last_length, d_quality_length + num_records -1, sizeof(int), cudaMemcpyDeviceToHost));
    // But in exclusive_scan, d_offset[num_records] = total_scores = sum(l0..ln-1)
    // Thus, better to use thrust::reduce

    // Using thrust to compute total_scores
    thrust::device_ptr<const int> d_len_ptr(d_quality_length);
    total_scores = thrust::reduce(d_len_ptr, d_len_ptr + num_records, 0, thrust::plus<int>());

    outMeta.total_scores = static_cast<size_t>(total_scores);

    // 4) Allocate device memory for compressed quality scores
    if (outMeta.total_scores > 0) {
        CUDA_CHECK(cudaMalloc(&outMeta.d_compressed, outMeta.total_scores * sizeof(char)));
        CUDA_CHECK(cudaMemset(outMeta.d_compressed, 0, outMeta.total_scores * sizeof(char)));
    } else {
        outMeta.d_compressed = nullptr; // Explicitly set to nullptr if no scores
    }

    // 5) Launch kernel to copy quality scores into contiguous memory only if there are scores to copy
    if (outMeta.total_scores > 0 && num_records > 0) {
        int blockSize = 256;
        int gridSize = (num_records + blockSize - 1) / blockSize;

        copyQualityScoresKernel<<<gridSize, blockSize>>>(
                d_data,
                d_fields_indices,
                d_quality_length,
                d_offset,
                num_records,
                outMeta.d_compressed
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 6) Cleanup intermediate device memory
    CUDA_CHECK(cudaFree(d_quality_length));
    CUDA_CHECK(cudaFree(d_offset));
}
