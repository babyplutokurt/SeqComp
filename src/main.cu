#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <chrono>

// Project headers
#include "../src/utils/file_utils.h"
#include "../include/indexing.h"
#include "../include/identifier_processing.h"
#include "../include/identifier_parser.h"
#include "../include/identifier_compressor.h"
#include "../include/gpulz.h"  // for gpulzCompress, gpulzDecompress

#include "cusz/type.h"
#include "hfclass.hh"

// Basecall compression headers
#include "../include/basecall_compress.h"
// Quality scores compression headers
#include "../include/quality_scores_process.h"

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(-1);                                             \
        }                                                         \
    } while(0)

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <fastq_file>\n";
        return 1;
    }
    const std::string file_path = argv[1];

    // 1) Read FASTQ to host
    std::vector<char> h_data = load_file_to_memory(file_path);
    if (h_data.empty()) {
        std::cerr << "Error: input file is empty or failed to read.\n";
        return 1;
    }
    std::cout << "Loaded file: " << file_path << " (size: "
              << h_data.size() << " bytes)\n";

    // 2) Transfer to device & build indices
    thrust::device_vector<char> d_data(h_data.begin(), h_data.end());

    int row_count = count_rows_fastq(d_data);
    if (row_count % 4 != 0) {
        std::cerr << "Error: FASTQ must have multiples of 4 lines.\n";
        return 1;
    }
    size_t num_records = row_count / 4;
    std::cout << "Detected " << row_count << " lines => "
              << num_records << " records.\n";

    thrust::device_vector<int> d_fields_indices;
    indexing_fields(d_data, row_count, d_fields_indices);

    // 3) Copy indices back for the FIRST identifier
    std::vector<int> h_fields_indices(d_fields_indices.size());
    thrust::copy(d_fields_indices.begin(), d_fields_indices.end(), h_fields_indices.begin());

    // Extract the first identifier
    size_t start = h_fields_indices[0];
    size_t end   = h_fields_indices[1] - 1;  // Exclude newline
    if (end > h_data.size()) {
        std::cerr << "Indexing error: end > file size\n";
        return 1;
    }
    std::string first_identifier(&h_data[start], &h_data[end]);

    // 4) Parse the first identifier
    ParsedIdentifier pid = parse_identifier(first_identifier, ". :/");
    std::cout << "First identifier has " << pid.segments.size() << " segments.\n"
              << "  #String segs: " << pid.num_string_segments
              << ", #Numeric segs: " << pid.num_numeric_segments << "\n";

    // 5) Allocate device memory for kernel output
    char* d_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buffer, h_data.size()));
    CUDA_CHECK(cudaMemcpy(d_buffer, h_data.data(), h_data.size(), cudaMemcpyHostToDevice));

    int* d_fields_indices_ptr = thrust::raw_pointer_cast(d_fields_indices.data());

    // Send segment_types and max_string_lengths to device
    thrust::device_vector<int> d_segment_types(pid.segment_types.begin(), pid.segment_types.end());
    int* d_segment_types_ptr = thrust::raw_pointer_cast(d_segment_types.data());

    thrust::device_vector<size_t> d_max_string_lengths(pid.max_string_lengths.begin(), pid.max_string_lengths.end());
    size_t* d_max_string_lengths_ptr = thrust::raw_pointer_cast(d_max_string_lengths.data());

    // Device arrays for numeric and string segments
    std::vector<int*> d_numeric_arrays(pid.num_numeric_segments, nullptr);
    int** d_numeric_segments = nullptr;
    CUDA_CHECK(cudaMalloc(&d_numeric_segments, pid.num_numeric_segments * sizeof(int*)));
    for (int i = 0; i < pid.num_numeric_segments; ++i) {
        CUDA_CHECK(cudaMalloc(&d_numeric_arrays[i], num_records * sizeof(int)));
    }
    CUDA_CHECK(cudaMemcpy(d_numeric_segments,
                          d_numeric_arrays.data(),
                          pid.num_numeric_segments * sizeof(int*),
                          cudaMemcpyHostToDevice));

    std::vector<char*> d_string_arrays(pid.num_string_segments, nullptr);
    char** d_string_segments = nullptr;
    CUDA_CHECK(cudaMalloc(&d_string_segments, pid.num_string_segments * sizeof(char*)));
    {
        int s_idx = 0;
        for (size_t seg_idx = 0; seg_idx < pid.segment_types.size(); seg_idx++) {
            if (pid.segment_types[seg_idx] == 0) {  // STRING
                size_t max_len = pid.max_string_lengths[s_idx] + 1;  // +1 for null terminator if needed
                CUDA_CHECK(cudaMalloc(&d_string_arrays[s_idx], num_records * max_len * sizeof(char)));
                s_idx++;
            }
        }
    }
    CUDA_CHECK(cudaMemcpy(d_string_segments,
                          d_string_arrays.data(),
                          pid.num_string_segments * sizeof(char*),
                          cudaMemcpyHostToDevice));

    auto start_count = std::chrono::high_resolution_clock::now();

    // 6) Launch process_identifiers kernel
    {
        int threads_per_block = 256;
        int blocks = static_cast<int>((num_records + threads_per_block - 1) / threads_per_block);

        process_identifiers<<<blocks, threads_per_block>>>(
                d_buffer,
                d_fields_indices_ptr,
                num_records,
                static_cast<int>(pid.segments.size()),
                d_segment_types_ptr,
                d_max_string_lengths_ptr,
                d_string_segments,
                d_numeric_segments
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 7) Sampling on device => constant/variable
    sample_segments_on_device(
            pid,
            num_records,
            d_string_arrays,
            d_numeric_arrays
    );

    // Print out which are constant vs. variable
    std::cout << "\nSegmentation details after sampling:\n";
    for (size_t i = 0; i < pid.segments.size(); i++) {
        const char* type_str   = (pid.segment_types[i] == 0) ? "STRING" : "NUMERIC";
        const char* static_str = pid.segment_static[i] ? "CONSTANT" : "VARIABLE";

        std::cout << "  Segment[" << i << "]:"
                  << " text=\"" << pid.segments[i] << "\""
                  << ", delimiter=\"" << pid.delimiters[i] << "\""
                  << ", type=" << type_str
                  << ", " << static_str
                  << "\n";
    }

    // 8) Compress each segment
    CompressedIdentifier cident = compress_identifier(
            pid,
            num_records,
            d_string_arrays,
            d_numeric_arrays
    );

    std::cout << "\n=== Compressed Results ===\n";
    for (size_t i = 0; i < cident.segments.size(); i++) {
        const CompressedSegment &cseg = cident.segments[i];

        // Basic info
        const char* seg_type = (cseg.segment_type == 0) ? "STRING" : "NUMERIC";
        const char* seg_var  = cseg.is_static ? "STATIC" : "VARIABLE";
        // If numeric & variable & !is_compressed => "Original"
        // else "Compressed"
        const bool  is_numeric_var_no_compress =
                (cseg.segment_type == 1 && !cseg.is_static && !cseg.is_compressed);
        const char* seg_isCompressed = is_numeric_var_no_compress
                                       ? "Original" : "Compressed";

        std::cout << " Segment[" << i << "]: type=" << seg_type
                  << ", " << seg_var
                  << ", " << seg_isCompressed
                  << ", delimiter=\"" << cseg.delimiter << "\""
                  << ", compressed_size=" << cseg.compressed_size
                  << "\n";

        if (cseg.is_static) {
            // For a static segment:
            if (cseg.segment_type == 0) {
                // string
                std::cout << "   => single string value: \""
                          << cseg.static_string_value << "\"\n";
            } else {
                // numeric
                std::cout << "   => single int value: "
                          << cseg.static_numeric_value << "\n";
            }
        } else {
            // For a variable segment:
            if (cseg.segment_type == 0) {
                // Possibly GPULZ
                if (cseg.is_compressed && cseg.is_gpulz) {
                    // The final size is cseg.compressed_size
                    // The device pointers are cseg.gpuLZData.{...}
                    std::cout << "   => GPU-LZ compressed on device. totalFlag+Data="
                              << cseg.compressed_size << " bytes (device)\n";
                }
                else {
                    // e.g., we stored the original or not implemented
                    std::cout << "   => string_compressed_data.size()="
                              << cseg.string_compressed_data.size()
                              << "\n";
                }
            } else {
                // numeric
                if (!cseg.is_compressed) {
                    std::cout << "   => stored original array, size="
                              << cseg.numeric_compressed_data.size()
                              << " bytes\n";
                } else {
                    std::cout << "   => numeric lorenzo+RLE data size="
                              << cseg.numeric_compressed_data.size()
                              << " bytes\n";
                }
            }
        }
    }

    auto end_count = std::chrono::high_resolution_clock::now();
    std::cout << "Identifier Compressed logic: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_count - start_count).count()
              << " ms" << std::endl;

    /*************************************************************************
     * 9) Basecall Compression
     *************************************************************************/
    std::cout << "\n=== Starting Basecall Compression ===\n";

    // Initialize BasecallCompressMeta structure
    BasecallCompressMeta basecallMeta;

    // Convert thrust::device_vector<char> -> raw pointer
    const char *d_data_ptr = thrust::raw_pointer_cast(d_data.data());
    const int  *d_fields_ptr = thrust::raw_pointer_cast(d_fields_indices.data());

    // Perform basecall compression
    basecallCompress(
            d_data_ptr,          // entire FASTQ data on device
            d_fields_ptr,        // line offsets
            row_count,           // total lines
            static_cast<int>(num_records),
            basecallMeta         // output metadata
    );

    std::cout << "Basecall compression done.\n"
              << "  total_bits=" << basecallMeta.total_bits
              << ", total_bytes=" << basecallMeta.total_bytes
              << ", N-count=" << basecallMeta.n_count << "\n";

    /*************************************************************************
     * 10) Quality Scores Compression
     *************************************************************************/
    std::cout << "\n=== Starting Quality Scores Compression ===\n";

    // Initialize QualityScoresCompressMeta structure
    QualityScoresCompressMeta qualityMeta;

    // Perform quality scores compression
    qualityScoresCompress(
            d_data_ptr,          // entire FASTQ data on device
            d_fields_ptr,        // line offsets
            row_count,           // total lines
            static_cast<int>(num_records),
            qualityMeta          // output metadata
    );




    std::cout << "Quality Scores compression done.\n"
              << "  total_scores=" << qualityMeta.total_scores
              << ", compressed_size=" << qualityMeta.total_scores * sizeof(char)
              << "\n";

    /*************************************************************************
     * 11) (Optional) Copy Compressed Quality Scores Back to Host for Verification
     *************************************************************************/

    std::vector<char> h_compressed_quality(qualityMeta.total_scores);
    CUDA_CHECK(cudaMemcpy(h_compressed_quality.data(), qualityMeta.d_compressed, qualityMeta.total_scores * sizeof(char), cudaMemcpyDeviceToHost));

    // Print the first 50 quality scores as a simple verification
    size_t printCount = std::min<size_t>(50, qualityMeta.total_scores);
    std::cout << "\nFirst " << printCount << " quality scores of compressed data:\n";
    for (size_t i = 0; i < printCount; i++) {
        std::cout << h_compressed_quality[i];
    }
    std::cout << "\n";

    /*************************************************************************
     * 12) Cleanup basecallMeta and qualityMeta device memory
     *************************************************************************/
    if (basecallMeta.d_compressed != nullptr) {
        CUDA_CHECK(cudaFree(basecallMeta.d_compressed));
    }
    if (basecallMeta.d_nPositions != nullptr) {
        CUDA_CHECK(cudaFree(basecallMeta.d_nPositions));
    }

    if (qualityMeta.d_compressed != nullptr) {
        CUDA_CHECK(cudaFree(qualityMeta.d_compressed));
    }

    /*************************************************************************
     * 13) Example: Copy identifier segments back to host (if needed for verification)
     *************************************************************************/
    // e.g., copy original numeric arrays for debug
    std::vector<int*> numeric_segments_host(pid.num_numeric_segments, nullptr);
    for (int i = 0; i < pid.num_numeric_segments; ++i) {
        numeric_segments_host[i] = new int[num_records];
        CUDA_CHECK(cudaMemcpy(numeric_segments_host[i],
                              d_numeric_arrays[i],
                              num_records * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    // e.g., copy original string arrays for debug
    std::vector<char*> string_segments_host(pid.num_string_segments, nullptr);
    {
        int s_idx = 0;
        for (size_t seg_idx = 0; seg_idx < pid.segment_types.size(); seg_idx++) {
            if (pid.segment_types[seg_idx] == 0) {  // STRING
                size_t max_len = pid.max_string_lengths[s_idx] + 1;
                string_segments_host[s_idx] = new char[num_records * max_len];
                CUDA_CHECK(cudaMemcpy(string_segments_host[s_idx],
                                      d_string_arrays[s_idx],
                                      num_records * max_len * sizeof(char),
                                      cudaMemcpyDeviceToHost));
                s_idx++;
            }
        }
    }

    /*************************************************************************
     * 14) (Optional) If you performed any GPULZ decompression earlier, it's already handled above
     *************************************************************************/

    // 15) Cleanup device memory
    CUDA_CHECK(cudaFree(d_buffer));
    for (int i = 0; i < pid.num_numeric_segments; ++i) {
        CUDA_CHECK(cudaFree(d_numeric_arrays[i]));
    }
    CUDA_CHECK(cudaFree(d_numeric_segments));
    for (int i = 0; i < pid.num_string_segments; ++i) {
        CUDA_CHECK(cudaFree(d_string_arrays[i]));
    }
    CUDA_CHECK(cudaFree(d_string_segments));

    // Cleanup host memory
    for (int i = 0; i < pid.num_numeric_segments; ++i) {
        delete[] numeric_segments_host[i];
    }
    for (int i = 0; i < pid.num_string_segments; ++i) {
        delete[] string_segments_host[i];
    }

    std::cout << "\nDone.\n";
    return 0;
}
