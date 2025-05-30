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
#include "../include/gpulz.h"
#include "../include/huffman_decode.h"
#include "../include/basecall_compress.h"
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


int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <fastq_file>\n";
        return 1;
    }
    const std::string file_path = argv[1];

    size_t identifier_compressed_total = 0;
    size_t basecalls_compressed_total = 0;
    size_t qualityScores_compressed_total = 0;
    size_t indices_compressed_total = 0;
    size_t original_data_total = 0;

    // 1) Read FASTQ to host
    std::vector<char> h_data = load_file_to_memory(file_path);
    if (h_data.empty()) {
        std::cerr << "Error: input file is empty or failed to read.\n";
        return 1;
    }
    std::cout << "Loaded file: " << file_path << " (size: "
              << h_data.size() << " bytes)\n";
    original_data_total = h_data.size();

    // 2) Transfer to device & build indices


    thrust::device_vector<char> d_data(h_data.begin(), h_data.end());

    std::cout << "\n=== Starting Indexing ===\n";

    auto start_time_indexing = std::chrono::high_resolution_clock::now();

    int row_count = count_rows_fastq(d_data);
    if (row_count % 4 != 0) {
        std::cerr << "Error: FASTQ must have multiples of 4 lines.\n";
        return 1;
    }
    size_t num_records = row_count / 4;

    thrust::device_vector<int> d_fields_indices;
    indexing_fields(d_data, row_count, d_fields_indices);
    indices_compressed_total += d_fields_indices.size();

    auto end_time_indexing = std::chrono::high_resolution_clock::now();
    auto time_indexing = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time_indexing - start_time_indexing).count();

    std::cout << "Indexing time: " << time_indexing << " ms\n";

    std::cout << "\n=== Starting Identifier Compression ===\n";
    auto start_time_identifier = std::chrono::high_resolution_clock::now();

    // 3) Copy indices back for the FIRST identifier
    std::vector<int> h_fields_indices(d_fields_indices.size());
    thrust::copy(d_fields_indices.begin(), d_fields_indices.end(), h_fields_indices.begin());

    // Extract the first identifier
    size_t start = h_fields_indices[0];
    size_t end = h_fields_indices[1] - 1;  // Exclude newline
    if (end > h_data.size()) {
        std::cerr << "Indexing error: end > file size\n";
        return 1;
    }
    std::string first_identifier(&h_data[start], &h_data[end]);

    // 4) Parse the first identifier
    ParsedIdentifier pid = parse_identifier(first_identifier, ". :/");

    // 5) Allocate device memory for kernel output
    char *d_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buffer, h_data.size()));
    CUDA_CHECK(cudaMemcpy(d_buffer, h_data.data(), h_data.size(), cudaMemcpyHostToDevice));

    int *d_fields_indices_ptr = thrust::raw_pointer_cast(d_fields_indices.data());

    // Send segment_types and max_string_lengths to device
    thrust::device_vector<int> d_segment_types(pid.segment_types.begin(), pid.segment_types.end());
    int *d_segment_types_ptr = thrust::raw_pointer_cast(d_segment_types.data());

    thrust::device_vector <size_t> d_max_string_lengths(pid.max_string_lengths.begin(), pid.max_string_lengths.end());
    size_t *d_max_string_lengths_ptr = thrust::raw_pointer_cast(d_max_string_lengths.data());

    // Device arrays for numeric and string segments
    std::vector<int *> d_numeric_arrays(pid.num_numeric_segments, nullptr);
    int **d_numeric_segments = nullptr;
    CUDA_CHECK(cudaMalloc(&d_numeric_segments, pid.num_numeric_segments * sizeof(int *)));
    for (int i = 0; i < pid.num_numeric_segments; ++i) {
        CUDA_CHECK(cudaMalloc(&d_numeric_arrays[i], num_records * sizeof(int)));
    }
    CUDA_CHECK(cudaMemcpy(d_numeric_segments,
                          d_numeric_arrays.data(),
                          pid.num_numeric_segments * sizeof(int *),
                          cudaMemcpyHostToDevice));

    std::vector<char *> d_string_arrays(pid.num_string_segments, nullptr);
    char **d_string_segments = nullptr;
    CUDA_CHECK(cudaMalloc(&d_string_segments, pid.num_string_segments * sizeof(char *)));
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
                          pid.num_string_segments * sizeof(char *),
                          cudaMemcpyHostToDevice));


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

    // 8) Compress each segment
    CompressedIdentifier cident = compress_identifier(
            pid,
            num_records,
            d_string_arrays,
            d_numeric_arrays
    );

    for (size_t i = 0; i < cident.segments.size(); i++) {
        const CompressedSegment &cseg = cident.segments[i];
        identifier_compressed_total += cseg.compressed_size;
    }

    auto end_time_identifier = std::chrono::high_resolution_clock::now();
    auto time_identifier = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time_identifier - start_time_identifier).count();
    std::cout << "Identifier Compression time: " << time_identifier << " ms\n";

    /*************************************************************************
     * 9) Basecall Compression
     *************************************************************************/
    std::cout << "\n=== Starting Basecall Compression ===\n";

    auto start_time_basecalls = std::chrono::high_resolution_clock::now();

    // Initialize BasecallCompressMeta structure
    BasecallCompressMeta basecallMeta;

    // Convert thrust::device_vector<char> -> raw pointer
    const char *d_basecall_data_ptr = thrust::raw_pointer_cast(d_data.data());
    const int *d_basecall_fields_ptr = thrust::raw_pointer_cast(d_fields_indices.data());

    // Perform basecall compression
    basecallCompress(
            d_basecall_data_ptr,          // entire FASTQ data on device
            d_basecall_fields_ptr,        // line offsets
            row_count,                     // total lines
            static_cast<int>(num_records),
            basecallMeta                   // output metadata
    );

    basecalls_compressed_total += basecallMeta.total_bytes + sizeof(int) * basecallMeta.n_count;


    auto end_time_basecalls = std::chrono::high_resolution_clock::now();
    auto time_basecalls = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time_basecalls - start_time_basecalls).count();
    std::cout << "Basecalls Compression time: " << time_basecalls << " ms\n";

    /*************************************************************************
     * 10) Quality Scores Compression
     *************************************************************************/
    std::cout << "\n=== Starting Quality Scores Compression ===\n";

    auto start_time_quality_preprocess = std::chrono::high_resolution_clock::now();

    // Initialize QualityScoresCompressMeta structure
    QualityScoresCompressMeta qualityMeta;

    // Perform quality scores compression
    qualityScoresCompress(
            d_basecall_data_ptr,          // entire FASTQ data on device
            d_fields_indices_ptr,        // line offsets
            row_count,                     // total lines
            static_cast<int>(num_records),
            h_data.size(),                 // total data size in bytes
            qualityMeta                    // output metadata
    );


    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    float sample_fraction = 0.2;
    size_t data_sample_size = qualityMeta.total_scores * sample_fraction;
    if (data_sample_size < 1) data_sample_size = 1;

    // Allocate memory to store the frequency of each character
    unsigned int *d_freq;
    CUDA_CHECK(cudaMalloc(&d_freq, 256 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_freq, 0, 256 * sizeof(unsigned int)));

    launch_histogram256(qualityMeta.d_compressed, d_freq, data_sample_size, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    add_one_kernel<<<1, 256, 0, stream>>>(d_freq);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto end_time_quality_preprocess = std::chrono::high_resolution_clock::now();
    auto time_quality_preprocess = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time_quality_preprocess - start_time_quality_preprocess).count();

    auto start_time_quality_buildbook = std::chrono::high_resolution_clock::now();

    size_t inlen = qualityMeta.total_scores;
    int bklen = 256;
    int pardeg = 4096;
    bool debug = false;
    std::cout << "Building Huffman Book...\n";
    phf::HuffmanCodec<u1, true> codec(inlen, bklen, pardeg, debug);
    codec.buildbook(reinterpret_cast<u4 *>(d_freq), reinterpret_cast<phf_stream_t>(stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto end_time_quality_buildbook = std::chrono::high_resolution_clock::now();
    auto time_quality_buildbook = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time_quality_buildbook - start_time_quality_buildbook).count();


    auto start_time_quality_encode = std::chrono::high_resolution_clock::now();

    PHF_BYTE *d_compressed_huff = nullptr;
    size_t compressed_len_huff = 0;
    codec.encode(
            thrust::raw_pointer_cast(qualityMeta.d_compressed), // input GPU pointer
            inlen,                          // input size (full data)
            &d_compressed_huff,                  // output GPU pointer (allocated by encode)
            &compressed_len_huff,                // output size
            stream
    );
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto end_time_quality_encode = std::chrono::high_resolution_clock::now();
    auto time_quality_encode = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time_quality_encode - start_time_quality_encode).count();
    qualityScores_compressed_total += compressed_len_huff;

    std::cout << "Quality Scores preprocess time: " << time_quality_preprocess << " ms\n";
    std::cout << "Quality Scores Build Book time: " << time_quality_buildbook << " ms\n";
    std::cout << "Quality Scores Encode time: " << time_quality_encode << " ms\n";

    /*************************************************************************
     * 11) Print out compression ratio and thrpughput
     *************************************************************************/
    size_t total_compressed_size = identifier_compressed_total + basecalls_compressed_total +
                                   qualityScores_compressed_total + 4;
    double compression_ratio = static_cast<double>(original_data_total) / static_cast<double>(total_compressed_size);
    std::cout << "\n=== Compression Metrics ===\n";
    std::cout << "Original Size: " << original_data_total << " bytes\n";
    std::cout << "Compressed Identifiers Size: " << identifier_compressed_total << " bytes\n";
    std::cout << "Compressed Basecalls Size: " << basecalls_compressed_total << " bytes\n";
    std::cout << "Compressed Quality Scores Size: " << qualityScores_compressed_total << " bytes\n";
    std::cout << "Compressed Indices Size: " << indices_compressed_total << " bytes\n";
    std::cout << "Total Compressed Size: " << total_compressed_size << " bytes\n";
    std::cout << "Compression Ratio (Original / Compressed): " << compression_ratio << "\n";

    std::cout << "\n=== Throughput Metrics ===\n";

    double throughput_bytes_per_sec = static_cast<double>(original_data_total) * 1000.0 /
                                      (time_indexing + time_identifier + time_basecalls + time_quality_preprocess +
                                       time_quality_encode);
    double throughput_MB_per_sec = throughput_bytes_per_sec / (1024.0 * 1024.0);

    std::cout << "Compression Throughput: "
              << throughput_MB_per_sec << " MB/sec)" << std::endl;

    /*************************************************************************
     * 12) Cleanup compressed meta device memory
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
     * 14) Cleanup device memory
     *************************************************************************/
    CUDA_CHECK(cudaFree(d_buffer));
    for (int i = 0; i < pid.num_numeric_segments; ++i) {
        CUDA_CHECK(cudaFree(d_numeric_arrays[i]));
    }
    CUDA_CHECK(cudaFree(d_numeric_segments));
    for (int i = 0; i < pid.num_string_segments; ++i) {
        CUDA_CHECK(cudaFree(d_string_arrays[i]));
    }
    CUDA_CHECK(cudaFree(d_string_segments));

    std::cout << "\nDone.\n";
    return 0;
}
