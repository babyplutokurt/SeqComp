#include "quality_scores_process.h"
#include "indexing.h"  // Ensure this header is correctly included and paths are set
#include <thrust/device_ptr.h>
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
 * 1) calcQualityScoreLineLengthsKernel
 *    quality_length[n] = (fields_indices[4n+4] - 1) - fields_indices[4n+3]
 *    Handles the last quality score line by using d_data_size as the end.
 ******************************************************************************/
__global__ void calcQualityScoreLineLengthsKernel(
        const int *d_fields_indices,
        int *d_quality_length,
        int num_records,
        int d_data_size  // Total size of the data buffer
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_records) {

        // Quality score line => 4*idx + 3
        int quality_idx = 4 * idx + 3;

        // Determine start position
        int start = d_fields_indices[quality_idx];

        int end;
        if (idx == num_records - 1) {
            // Last quality score line: use d_data_size as end
            end = d_data_size - 1;
        } else {
            // Normal case: Use next index
            end = d_fields_indices[quality_idx + 1] - 1; // Exclude newline
        }

        // Ensure that end >= start to prevent negative lengths
        int length = (end >= start) ? (end - start) : 0;
        d_quality_length[idx] = length;
    }
}



/******************************************************************************
 * 2) calcQualityScoreLengthsFromFields
 *    Host function to launch the kernel for calculating quality score lengths.
 ******************************************************************************/
void calcQualityScoreLengthsFromFields(
        const int *d_fields_indices,
        int *d_quality_length,
        int num_records,
        int d_data_size  // Total size of the data buffer
) {
    int blockSize = 256;
    int gridSize = (num_records + blockSize - 1) / blockSize;

    calcQualityScoreLineLengthsKernel<<<gridSize, blockSize>>>(
            d_fields_indices,
            d_quality_length,
            num_records,
            d_data_size
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/******************************************************************************
 * 3) calcQualityScoreOffsets
 *    exclusive scan of quality_length => offset array
 *    d_offset has size (num_records + 1)
 ******************************************************************************/
void calcQualityScoreOffsets(
        const int *d_quality_length,
        int num_records,
        int *d_offset
) {
    // Ensure the first element is set to zero (starting point for prefix sum)
    CUDA_CHECK(cudaMemset(d_offset, 0, sizeof(int)));

    // Perform an exclusive scan starting from d_offset + 1
    thrust::device_ptr<const int> d_len_ptr(d_quality_length);
    thrust::device_ptr<int> d_offset_ptr(d_offset + 1);  // Start writing from index 1

    thrust::inclusive_scan(d_len_ptr, d_len_ptr + num_records, d_offset_ptr);
}


/******************************************************************************
 * 4) copyQualityScoresKernel
 *    Kernel to copy quality scores into a contiguous device array based on prefix offsets
 ******************************************************************************/
__global__ void copyQualityScoresKernel(
        const char *d_data,
        const int *d_fields_indices,
        const int *d_quality_length,
        const int *d_offset,
        int num_records,
        unsigned char *d_compressed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_records) return;

    // Quality score line => 4*idx + 3
    int quality_idx = 4 * idx + 3;
    int start = d_fields_indices[quality_idx];
    int length = d_quality_length[idx];
    int offset = d_offset[idx]; // Starting position in d_compressed

    // Copy the quality scores to the compressed array
    for (int i = 0; i < length; ++i) {
        d_compressed[offset + i] = d_data[start + i];
    }
}

/******************************************************************************
 * 5) qualityScoresCompress
 *    Complete compression process for quality scores
 ******************************************************************************/
void qualityScoresCompress(
        const char *d_data,
        const int *d_fields_indices,
        int row_count,
        int num_records,
        int d_data_size,                // Total data size in bytes
        QualityScoresCompressMeta &outMeta
) {
    // 1) Calculate quality score lengths
    int *d_quality_length = nullptr;
    CUDA_CHECK(cudaMalloc(&d_quality_length, num_records * sizeof(int)));
    calcQualityScoreLengthsFromFields(d_fields_indices, d_quality_length, num_records, d_data_size);

    // 2) Calculate prefix sum offsets using exclusive scan
    // Allocate d_offset with num_records + 1 elements
    int *d_offset = nullptr;
    CUDA_CHECK(cudaMalloc(&d_offset, (num_records + 1) * sizeof(int))); // exclusive scan result size = num_records + 1
    // Initialize d_offset[0] = 0
    CUDA_CHECK(cudaMemset(d_offset, 0, sizeof(int)));
    calcQualityScoreOffsets(d_quality_length, num_records, d_offset);

    // 3) Retrieve total_scores from d_offset[num_records]
    int total_scores = 0;
    CUDA_CHECK(cudaMemcpy(&total_scores, d_offset + num_records, sizeof(int), cudaMemcpyDeviceToHost));

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
}
