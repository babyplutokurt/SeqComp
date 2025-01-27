#pragma once

#include <cstddef>
#include <cstdint>

/**
 * @brief Metadata describing compressed basecalls:
 */
struct BasecallCompressMeta
{
    uint8_t *d_compressed = nullptr;  // 2-bit array in device memory
    size_t   total_bits   = 0;
    size_t   total_bytes  = 0;

    int   *d_nPositions   = nullptr;  // (record, pos_in_line) pairs
    int    n_count        = 0;
};

/**
 * @brief Kernel function to fill basecall lengths from d_fields_indices:
 *        basecall_length[n] = (fields_indices[4n+2] - 1) - fields_indices[4n+1].
 * We expose it as a function for clarity.
 *
 * @param d_fields_indices    device array (row_count+1)
 * @param d_basecall_length   output device array (num_records)
 * @param num_records
 */
void calcBasecallLengthsFromFields(
        const int *d_fields_indices,
        int *d_basecall_length,
        int num_records
);

/**
 * @brief Compute prefix sum offsets from basecall_length array on the device.
 *        d_offset has size (num_records+1). The last entry is the sum of all lengths.
 *
 * @param d_basecall_length  device array (num_records)
 * @param num_records
 * @param d_offset           device array (num_records+1)
 */
void calcBasecallOffsets(
        const int *d_basecall_length,
        int num_records,
        int *d_offset
);

/**
 * @brief 2-bit compress the basecall lines (lines 4n+1).
 *        - We first get basecall_length from d_fields_indices
 *        - Then we prefix-sum to get basecall_offset
 *        - Then we allocate the final compressed array & N positions
 *        - compress each record in a kernel
 *
 * @param d_data            full FASTQ device pointer
 * @param d_fields_indices  line offsets
 * @param row_count
 * @param num_records
 * @param outMeta
 */
void basecallCompress(
        const char   *d_data,
        const int    *d_fields_indices,
        int           row_count,
        int           num_records,
        BasecallCompressMeta &outMeta
);

/**
 * @brief Decompress the 2-bit array in outMeta to d_output in chunkSize increments,
 *        then restore 'N' positions.
 *
 * @param outMeta
 * @param d_output
 * @param total_basecalls
 * @param chunkSizeBytes
 * @param d_fields_indices
 */
void basecallDecompress(
        const BasecallCompressMeta &outMeta,
        char *d_output,
        size_t total_basecalls,
        int chunkSizeBytes,
        const int *d_fields_indices
);
