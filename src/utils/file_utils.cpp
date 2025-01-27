#include "file_utils.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

std::vector<char> load_file_to_memory(const std::string &file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    // Get the file size
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read file content into a vector
    std::vector<char> buffer(file_size);
    if (!file.read(buffer.data(), file_size)) {
        throw std::runtime_error("Failed to read file: " + file_path);
    }

    return buffer;
}
