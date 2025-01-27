#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <thrust/device_vector.h>

#include "identifier_parser.h"  // for ParsedIdentifier
#include "gpulz.h"

/**
 * @brief Data structure for a single compressed segment.
 *        Depending on segment_type and is_static, different fields will be relevant.
 */
struct CompressedSegment {
    int segment_type;       // 0=string, 1=numeric
    bool is_static;         // true => constant across all records
    bool is_compressed;     // true => used some compression (lorenzo+RLE or GPULZ)
    bool is_gpulz;          // true => specifically used GPULZ for string
    std::string delimiter;
    size_t compressed_size;

    int static_numeric_value;
    std::string static_string_value;

    // For variable numeric:
    std::vector<char> numeric_compressed_data;

    // For variable string (GPULZ or original):
    std::vector<char> string_compressed_data;

    // *** NEW: store a GpuLZMeta struct if using GPULZ for string data
    // Only relevant if `is_gpulz == true` and `is_compressed == true`.
    GpuLZMeta gpuLZData;

    CompressedSegment()
            : segment_type(0),
              is_static(false),
              is_compressed(false),
              is_gpulz(false),
              compressed_size(0),
              static_numeric_value(0)
    {}
};


/**
 * @brief Holds the compression results for the entire identifier
 *        (which can have multiple segments).
 */
struct CompressedIdentifier {
    std::vector<CompressedSegment> segments;
};

/**
 * @brief Main function to compress each segment of an identifier.
 *
 * @param pid                 [in] The parsed identifier (with segment info, static flags, etc.).
 * @param num_records         [in] Number of records in the entire FASTQ chunk.
 * @param d_string_arrays     [in] Device pointers for each string segment (columnar).
 * @param d_numeric_arrays    [in] Device pointers for each numeric segment (columnar).
 *
 * @return CompressedIdentifier containing all compressed segments.
 */
CompressedIdentifier compress_identifier(
        const ParsedIdentifier &pid,
        size_t num_records,
        const std::vector<char*> &d_string_arrays,
        const std::vector<int*> &d_numeric_arrays
);

