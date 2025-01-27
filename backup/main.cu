#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>  // Not strictly necessary now, but okay to keep

#include "../src/utils/file_utils.h"             // load_file_to_memory
#include "../include/indexing.h"                 // count_rows_fastq, indexing_fields
#include "../include/identifier_processing.h"    // process_identifiers kernel

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <fastq_file>\n";
        return 1;
    }
    std::string file_path = argv[1];

    // 1. Load file into host memory
    std::vector<char> h_data = load_file_to_memory(file_path);
    if (h_data.empty()) {
        std::cerr << "Error: input file is empty or failed to read.\n";
        return 1;
    }

    // 2. Transfer the data to device memory
    thrust::device_vector<char> d_data(h_data.begin(), h_data.end());

    // 3. Count rows & build indexing
    int row_count = count_rows_fastq(d_data);
    std::cout << "Detected " << row_count << " lines (newline-terminated) in the FASTQ.\n";

    // Build the line-start indices on the GPU
    thrust::device_vector<int> d_fields_indices;
    indexing_fields(d_data, row_count, d_fields_indices);
    // After this, d_fields_indices should have row_count+1 entries:
    // index of line 0, line 1, line 2..., plus an extra one for the final boundary.

    // 4. Determine how many records
    if (row_count % 4 != 0) {
        std::cerr << "Error: FASTQ should have multiples of 4 lines per record.\n";
        return 1;
    }
    size_t num_records = row_count / 4;
    std::cout << "Number of FASTQ records: " << num_records << "\n";

    // 5. Sample the first identifier to figure out segmentation
    //    We'll do this on the host for simplicity.
    //    But d_fields_indices is in device memory, so copy it back first:
    std::vector<int> h_fields_indices(d_fields_indices.size());
    thrust::copy(d_fields_indices.begin(), d_fields_indices.end(), h_fields_indices.begin());

    // The first record's identifier line is line 0 → h_fields_indices[0]
    // The next line's start is h_fields_indices[1], so the end of line 0 is (that - 1).
    size_t start = h_fields_indices[0];
    size_t end   = h_fields_indices[1] - 1; // skip the newline
    if (end > h_data.size()) {
        std::cerr << "Indexing error: end > file size\n";
        return 1;
    }
    size_t first_id_len = end - start;
    std::string first_identifier(&h_data[start], &h_data[start + first_id_len]);

    // Delimiters to match the device code
    const std::string delimiters = ". :/";

    // Let's parse the first identifier to define our segments
    std::vector<std::string> segments;
    std::vector<int> segment_types;     // 0=string, 1=numeric
    std::vector<size_t> max_string_lengths;

    {
        size_t pos = 0;
        while (pos < first_identifier.size()) {
            // find the next delimiter
            size_t dpos = first_identifier.find_first_of(delimiters, pos);
            if (dpos == std::string::npos) {
                dpos = first_identifier.size();
            }
            size_t seg_len = dpos - pos;
            if (seg_len > 0) {
                std::string seg = first_identifier.substr(pos, seg_len);
                segments.push_back(seg);

                // check numeric
                bool all_digits = true;
                for (char c : seg) {
                    if (!isdigit(static_cast<unsigned char>(c))) {
                        all_digits = false;
                        break;
                    }
                }
                segment_types.push_back(all_digits ? 1 : 0);
                if (!all_digits) {
                    max_string_lengths.push_back(seg_len);
                }
            }
            pos = (dpos < first_identifier.size()) ? dpos + 1 : dpos;
        }
    }

    int num_segments = segments.size();
    std::cout << "First identifier has " << num_segments << " segments.\n";
    std::cout << "(Segments: ";
    for (auto &seg : segments) {
        std::cout << seg << " | ";
    }
    std::cout << ")\n";

    // (Optional) If you want to refine max lengths by sampling more lines,
    // do so here the same way you did in your older code.

    // Count how many are string vs numeric
    int num_string_segments = 0;
    int num_numeric_segments = 0;
    for (int t : segment_types) {
        if (t == 0) num_string_segments++;
        else        num_numeric_segments++;
    }

    // 6. Allocate host arrays for final parsed results
    //    Each numeric segment is stored in an int[num_records].
    //    Each string segment is stored in a char[num_records * (max_len+1)].
    std::vector<int*> numeric_segments_host(num_numeric_segments);
    for (int i = 0; i < num_numeric_segments; ++i) {
        numeric_segments_host[i] = new int[num_records];
    }

    std::vector<char*> string_segments_host(num_string_segments, nullptr);
    {
        int s_index = 0;
        for (int seg_index = 0; seg_index < num_segments; ++seg_index) {
            if (segment_types[seg_index] == 0) {
                size_t max_len = max_string_lengths[s_index] + 1; // +1 for null terminator
                string_segments_host[s_index] = new char[num_records * max_len];
                memset(string_segments_host[s_index], 0, num_records * max_len);
                s_index++;
            }
        }
    }

    // 7. Allocate and copy data to device
    // (a) d_buffer: pointer to the entire FASTQ data
    char *d_buffer = nullptr;
    cudaMalloc(&d_buffer, h_data.size() * sizeof(char));
    cudaMemcpy(d_buffer, h_data.data(), h_data.size() * sizeof(char), cudaMemcpyHostToDevice);

    // (b) d_fields_indices: already in device_vector
    int *d_fields_indices_ptr = thrust::raw_pointer_cast(d_fields_indices.data());

    // (c) d_segment_types
    int *d_segment_types = nullptr;
    cudaMalloc(&d_segment_types, num_segments * sizeof(int));
    cudaMemcpy(d_segment_types, segment_types.data(), num_segments * sizeof(int), cudaMemcpyHostToDevice);

    // (d) d_max_string_lengths
    // We have one entry per string-type segment
    // Use standard C++ vector for the host, then copy to device_vector via Thrust
    std::vector<size_t> host_string_maxes(max_string_lengths.begin(), max_string_lengths.end());
    thrust::device_vector<size_t> d_max_string_lengths(host_string_maxes.begin(), host_string_maxes.end());
    size_t *d_max_string_lengths_ptr = thrust::raw_pointer_cast(d_max_string_lengths.data());

    // (e) numeric segments (array of pointers)
    int **d_numeric_segments = nullptr;
    cudaMalloc(&d_numeric_segments, num_numeric_segments * sizeof(int*));
    std::vector<int*> d_numeric_segments_array(num_numeric_segments, nullptr);
    for (int i = 0; i < num_numeric_segments; ++i) {
        cudaMalloc(&d_numeric_segments_array[i], num_records * sizeof(int));
    }
    cudaMemcpy(d_numeric_segments,
               d_numeric_segments_array.data(),
               num_numeric_segments * sizeof(int*),
               cudaMemcpyHostToDevice);

    // (f) string segments (array of pointers)
    char **d_string_segments = nullptr;
    cudaMalloc(&d_string_segments, num_string_segments * sizeof(char*));
    std::vector<char*> d_string_segments_array(num_string_segments, nullptr);
    {
        int s_index = 0;
        for (int seg_index = 0; seg_index < num_segments; ++seg_index) {
            if (segment_types[seg_index] == 0) {
                size_t max_len = max_string_lengths[s_index] + 1;
                cudaMalloc(&d_string_segments_array[s_index], num_records * max_len * sizeof(char));
                s_index++;
            }
        }
    }
    cudaMemcpy(d_string_segments,
               d_string_segments_array.data(),
               num_string_segments * sizeof(char*),
               cudaMemcpyHostToDevice);

    // 8. Kernel launch
    int threadsPerBlock = 256;
    int blocks = (num_records + threadsPerBlock - 1) / threadsPerBlock;

    process_identifiers<<<blocks, threadsPerBlock>>>(
            d_buffer,
            d_fields_indices_ptr,
            num_records,
            num_segments,
            d_segment_types,
            d_max_string_lengths_ptr,
            d_string_segments,
            d_numeric_segments
    );
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // 9. Copy results back
    // (a) numeric
    for (int i = 0; i < num_numeric_segments; ++i) {
        cudaMemcpy(numeric_segments_host[i],
                   d_numeric_segments_array[i],
                   num_records * sizeof(int),
                   cudaMemcpyDeviceToHost);
    }

    // (b) string
    {
        int s_index = 0;
        for (int seg_index = 0; seg_index < num_segments; ++seg_index) {
            if (segment_types[seg_index] == 0) {
                size_t max_len = max_string_lengths[s_index] + 1;
                cudaMemcpy(string_segments_host[s_index],
                           d_string_segments_array[s_index],
                           num_records * max_len * sizeof(char),
                           cudaMemcpyDeviceToHost);
                s_index++;
            }
        }
    }

    // Optional sanity-check: Print segments of a few records
    std::cout << "\nExample: Print first 3 records' identifier segments...\n";
    for (size_t r = 0; r < 3 && r < num_records; ++r) {
        std::cout << "Record " << r << ": \n";
        int s_idx = 0, n_idx = 0;
        for (int seg = 0; seg < num_segments; ++seg) {
            if (segment_types[seg] == 0) {
                // string
                size_t max_len = max_string_lengths[s_idx] + 1;
                char *seg_ptr = string_segments_host[s_idx] + (r * max_len);
                std::cout << "  String segment " << seg << ": " << seg_ptr << "\n";
                s_idx++;
            } else {
                // numeric
                int val = numeric_segments_host[n_idx][r];
                std::cout << "  Numeric segment " << seg << ": " << val << "\n";
                n_idx++;
            }
        }
    }

    // 10. Cleanup
    // Host
    for (int i = 0; i < num_numeric_segments; ++i) {
        delete[] numeric_segments_host[i];
    }
    {
        int s_index = 0;
        for (int seg_index = 0; seg_index < num_segments; ++seg_index) {
            if (segment_types[seg_index] == 0) {
                delete[] string_segments_host[s_index];
                s_index++;
            }
        }
    }

/*
    // 10) Print last 3 identifiers for debugging
    if (num_records == 0) {
        std::cout << "\nNo records found.\n";
    } else {
        size_t start_rec = (num_records > 3) ? (num_records - 3) : 0;
        std::cout << "\nLast 3 records:\n";
        // Build mapping from seg_idx -> s_idx or n_idx
        std::vector<int> seg2str(pid.segments.size(), -1), seg2num(pid.segments.size(), -1);
        {
            int s_idx = 0, n_idx = 0;
            for (size_t seg_idx = 0; seg_idx < pid.segments.size(); seg_idx++) {
                if (pid.segment_types[seg_idx] == 0) {
                    seg2str[seg_idx] = s_idx++;
                } else {
                    seg2num[seg_idx] = n_idx++;
                }
            }
        }

        for (size_t r = start_rec; r < num_records; ++r) {
            std::cout << "Record " << r << ":\n";
            for (size_t seg = 0; seg < pid.segments.size(); seg++) {
                if (pid.segment_types[seg] == 0) {
                    int s_idx_map = seg2str[seg];
                    size_t max_len = pid.max_string_lengths[s_idx_map] + 1;
                    char* seg_ptr = &string_segments_host[s_idx_map][r * max_len];
                    std::cout << "  String seg " << seg << ": " << seg_ptr << "\n";
                } else {
                    int n_idx_map = seg2num[seg];
                    int val = numeric_segments_host[n_idx_map][r];
                    std::cout << "  Numeric seg " << seg << ": " << val << "\n";
                }
            }
        }
    }
*/

    // Device
    cudaFree(d_buffer);
    cudaFree(d_segment_types);
    // d_fields_indices is in thrust::device_vector, so no direct cudaFree needed
    for (int i = 0; i < num_numeric_segments; ++i) {
        cudaFree(d_numeric_segments_array[i]);
    }
    cudaFree(d_numeric_segments);
    for (int i = 0; i < num_string_segments; ++i) {
        cudaFree(d_string_segments_array[i]);
    }
    cudaFree(d_string_segments);

    std::cout << "Done.\n";
    return 0;
}



