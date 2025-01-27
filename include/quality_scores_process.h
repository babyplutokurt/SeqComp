#ifndef QUALITY_SCORES_PROCESS_H
#define QUALITY_SCORES_PROCESS_H

#include <cstddef>
#include <cstdint>

/**
 * @brief Metadata describing compressed quality scores.
 */
struct QualityScoresCompressMeta
{
    char *d_compressed = nullptr;      // Contiguous array of quality scores in device memory
    size_t total_scores = 0;           // Total number of quality scores
};

/**
 * @brief Kernel function to calculate the lengths of quality score lines.
 *        quality_length[n] = (fields_indices[4n+4] - 1) - fields_indices[4n+3]
 *
 * @param d_fields_indices     Device array containing line start indices
 * @param d_quality_length     Output device array to store lengths of each quality score line
 * @param num_records          Number of FastQ records
 */
void calcQualityScoreLengthsFromFields(
        const int *d_fields_indices,
        int *d_quality_length,
        int num_records
);

/**
 * @brief Compute prefix sum offsets from quality_length array on the device.
 *        d_offset has size (num_records + 1). The last entry is the sum of all lengths.
 *
 * @param d_quality_length     Device array containing lengths of quality score lines
 * @param num_records          Number of FastQ records
 * @param d_offset             Device array to store prefix sum offsets
 */
void calcQualityScoreOffsets(
        const int *d_quality_length,
        int num_records,
        int *d_offset
);

/**
 * @brief Kernel to copy quality scores into a contiguous device array based on prefix offsets.
 *
 * @param d_data              Full FastQ data on device
 * @param d_fields_indices    Device array containing line start indices
 * @param d_quality_length    Device array containing lengths of quality score lines
 * @param d_offset            Device array containing prefix sum offsets
 * @param num_records         Number of FastQ records
 * @param d_compressed        Output device array to store contiguous quality scores
 */
__global__ void copyQualityScoresKernel(
        const char *d_data,
        const int *d_fields_indices,
        const int *d_quality_length,
        const int *d_offset,
        int num_records,
        char *d_compressed
);

/**
 * @brief Perform the entire quality score compression process:
 *        1. Calculate quality score lengths
 *        2. Compute prefix sum offsets
 *        3. Allocate device memory for compressed quality scores
 *        4. Launch kernel to copy quality scores into contiguous memory
 *
 * @param d_data              Full FastQ data on device
 * @param d_fields_indices    Device array containing line start indices
 * @param row_count           Total number of lines in FastQ file
 * @param num_records         Number of FastQ records
 * @param outMeta             Output metadata for compressed quality scores
 */
void qualityScoresCompress(
        const char *d_data,
        const int *d_fields_indices,
        int row_count,
        int num_records,
        QualityScoresCompressMeta &outMeta
);

#endif // QUALITY_SCORES_PROCESS_H
