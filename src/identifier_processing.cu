#include "identifier_processing.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define MAX_SEGMENTS 16
#define MAX_DELIMITERS 4

// Delimiters
__device__ const char delimiters[] = ". :/";

// Device function to check if a character is a delimiter
__device__ bool is_delimiter(char c) {
    for (int i = 0; i < MAX_DELIMITERS; ++i) {
        if (c == delimiters[i]) {
            return true;
        }
    }
    return false;
}

// Device function to convert a substring to an integer
__device__ bool string_to_int(const char* str, size_t length, int &result) {
    result = 0;
    for (size_t i = 0; i < length; ++i) {
        char c = str[i];
        if (c < '0' || c > '9') {
            return false; // Not numeric
        }
        result = result * 10 + (c - '0');
    }
    return true;
}

// Kernel function to process identifiers
__global__ void process_identifiers(
        char *d_buffer,
        const int *d_fields_indices, // line starts
        size_t num_records,
        int num_segments,
        int *d_segment_types,         // 0=string, 1=numeric
        size_t *d_max_string_lengths, // per string-segment
        char **d_string_segments,
        int **d_numeric_segments
)
{
    int record_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (record_idx >= num_records) return;

    // The identifier line is line 4*record_idx
    int line_idx = 4 * record_idx; // 4 lines per FASTQ record
    // Start of this line
    size_t start = d_fields_indices[line_idx];
    // Start of next line => end of the identifier line
    // We do -1 to skip newline
    size_t end   = d_fields_indices[line_idx + 1] - 1;
    size_t length = (end > start) ? (end - start) : 0;

    // The identifier string in device buffer
    char *identifier = d_buffer + start;

    int segment_index = 0;
    int string_segment_index = 0;
    int numeric_segment_index = 0;

    size_t segment_start = 0;
    size_t i = 0;

    while ((i <= length) && (segment_index < num_segments)) {
        // Safe check for boundary: at i == length, we treat as delimiter
        char c = (i < length) ? identifier[i] : '\0';
        if (is_delimiter(c) || c == '\0') {
            // End of a segment
            size_t seg_length = i - segment_start;
            if (seg_length > 0) {
                // Extract the segment into a temporary buffer
                char segment_buffer[256]; // assume <256
                size_t copy_len = (seg_length < 255) ? seg_length : 255;
                for (size_t j = 0; j < copy_len; ++j) {
                    segment_buffer[j] = identifier[segment_start + j];
                }
                segment_buffer[copy_len] = '\0';

                // String or numeric?
                if (d_segment_types[segment_index] == 0) {
                    // String segment
                    size_t max_len = d_max_string_lengths[string_segment_index] + 1;
                    char *dest = d_string_segments[string_segment_index] + (record_idx * max_len);
                    size_t dest_copy_len = (copy_len < (max_len - 1)) ? copy_len : (max_len - 1);
                    for (size_t j = 0; j < dest_copy_len; ++j) {
                        dest[j] = segment_buffer[j];
                    }
                    dest[dest_copy_len] = '\0';

                    string_segment_index++;
                }
                else {
                    // Numeric segment
                    int num = 0;
                    bool is_num = string_to_int(segment_buffer, copy_len, num);
                    if (is_num) {
                        d_numeric_segments[numeric_segment_index][record_idx] = num;
                    } else {
                        // If parsing fails, store -1 or handle error
                        d_numeric_segments[numeric_segment_index][record_idx] = -1;
                    }
                    numeric_segment_index++;
                }
            }

            segment_index++;
            segment_start = i + 1;
        }
        i++;
    }
}
