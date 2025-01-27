#include "utils/file_utils.h"
#include "../include/indexing.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <vector>
#include <iostream>
#include <chrono> // For timing

int main() {
    const std::string file_path = "/home/tus53997/SeqComp/data/SRR1295433_1_10000000.fastq";

    try {
        // Load FastQ file into host memory
        std::cout << "Loading file into host memory..." << std::endl;
        auto start_load = std::chrono::high_resolution_clock::now();
        std::vector<char> h_data = load_file_to_memory(file_path);
        auto end_load = std::chrono::high_resolution_clock::now();
        std::cout << "File loaded in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load).count()
                  << " ms" << std::endl;

        // Copy data to device memory
        thrust::device_vector<char> d_data(h_data.begin(), h_data.end());

        // Count rows
        std::cout << "Counting rows in FastQ file..." << std::endl;
        auto start_count = std::chrono::high_resolution_clock::now();
        int row_count = count_rows_fastq(d_data);
        auto end_count = std::chrono::high_resolution_clock::now();
        std::cout << "Number of rows: " << row_count << std::endl;
        std::cout << "Row counting completed in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_count - start_count).count()
                  << " ms" << std::endl;

        // Index newline characters
        std::cout << "Indexing newline characters..." << std::endl;
        thrust::device_vector<int> fields_indices;
        auto start_index = std::chrono::high_resolution_clock::now();
        indexing_fields(d_data, row_count, fields_indices);
        auto end_index = std::chrono::high_resolution_clock::now();
        std::cout << "Indexing completed in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_index - start_index).count()
                  << " ms" << std::endl;

        // Copy indices back to host
        std::vector<int> h_indices(row_count); // Allocate enough space
        thrust::copy(fields_indices.begin(), fields_indices.end(), h_indices.begin());

        // Print the first 10 indices
        std::cout << "First 10 indices of newline characters: ";
        for (int i = 0; i < std::min(10, row_count); ++i) {
            std::cout << h_indices[i] << " ";
        }
        std::cout << std::endl;

        // Print the Last 10 indices
        std::cout << "Last 10 indices of newline characters: ";
        for (int i = row_count - 10; i < std::min(row_count, row_count); ++i) {
            std::cout << h_indices[i] << " ";
        }
        std::cout << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
