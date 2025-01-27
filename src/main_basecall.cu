#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

// Project headers
#include "../src/utils/file_utils.h"           // load_file_to_memory
#include "../include/indexing.h"               // count_rows_fastq, indexing_fields
#include "../include/basecall_compress.h"      // basecallCompress, basecallDecompress

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <fastq_file>\n";
        return 1;
    }
    const std::string file_path = argv[1];

    // 1) Load FASTQ file into host memory
    std::vector<char> h_data = load_file_to_memory(file_path);
    if (h_data.empty()) {
        std::cerr << "Error: input file is empty or failed to read.\n";
        return 1;
    }
    std::cout << "Loaded file: " << file_path << " (size: "
              << h_data.size() << " bytes)\n";

    // 2) Transfer data to device & build line indices
    thrust::device_vector<char> d_data(h_data.begin(), h_data.end());

    // Count rows
    int row_count = count_rows_fastq(d_data);
    if (row_count % 4 != 0) {
        std::cerr << "Error: FASTQ must have multiples of 4 lines.\n";
        return 1;
    }
    size_t num_records = row_count / 4;
    std::cout << "Detected " << row_count << " lines => "
              << num_records << " records.\n";

    // Build line offsets
    thrust::device_vector<int> d_fields_indices;
    indexing_fields(d_data, row_count, d_fields_indices);

    // 3) Setup for basecall compression
    // We'll do it in place on the entire FASTQ device pointer (d_data)
    // Then basecallCompress will look at lines 4n+1 for each record's basecall line
    // row_count and d_fields_indices used

    // We'll fill a metadata struct describing final compressed data
    BasecallCompressMeta basecallMeta;

    // Convert thrust::device_vector<char> -> raw pointer
    const char *d_data_ptr = thrust::raw_pointer_cast(d_data.data());
    const int  *d_fields_ptr = thrust::raw_pointer_cast(d_fields_indices.data());

    // Actually compress
    basecallCompress(
            d_data_ptr,       // entire fastq data on device
            d_fields_ptr,
            row_count,        // total lines
            static_cast<int>(num_records),
            basecallMeta
    );

    std::cout << "\nBasecall compression done.\n"
              << "  total_bits=" << basecallMeta.total_bits
              << ", total_bytes=" << basecallMeta.total_bytes
              << ", N-count=" << basecallMeta.n_count << "\n";

    // 4) Decompress into a device array
    // We need the total basecalls.  We can do that as 2 * basecallMeta.total_bits / 2 => basecallMeta.total_bits/2
    // but let's be consistent with how we used an offset approach
    // We can do a simpler approach: we know the last record's basecall offset => basecallMeta.total_bits/2

    size_t total_basecalls = basecallMeta.total_bits / 2;
    // Allocate device array to hold the decompressed basecalls
    char *d_decompressed = nullptr;
    cudaMalloc(&d_decompressed, total_basecalls);

    // For chunk-based approach, let's pick chunkSize=1000
    int chunkSizeBytes = 1000;
    basecallDecompress(
            basecallMeta,
            d_decompressed,
            total_basecalls,
            chunkSizeBytes,
            d_fields_ptr   // needed for restoreN
    );

    std::cout << "Basecall decompression done.\n";

    // 5) Copy the decompressed data to host
    std::vector<char> h_decompressed(total_basecalls);
    cudaMemcpy(h_decompressed.data(), d_decompressed, total_basecalls, cudaMemcpyDeviceToHost);

    // 6) Print first 50 bytes of decompressed basecalls
    size_t printCount = std::min<size_t>(50, total_basecalls);
    std::cout << "\nFirst " << printCount << " bases of decompressed data:\n";
    for (size_t i = 0; i < printCount; i++) {
        std::cout << h_decompressed[i];
    }
    std::cout << "\n";

    // Cleanup
    cudaFree(d_decompressed);

    // free the basecallMeta arrays
    cudaFree(basecallMeta.d_compressed);
    cudaFree(basecallMeta.d_nPositions);

    std::cout << "\nDone.\n";
    return 0;
}
