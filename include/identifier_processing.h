#pragma once
#include <cstddef>  // for size_t

// Kernel from your old code to segment identifiers.
__global__ void process_identifiers(
        char *d_buffer,
        const int *d_fields_indices,
        size_t num_records,
        int num_segments,
        int *d_segment_types,           // 0=string, 1=numeric
        size_t *d_max_string_lengths,   // for string segments
        char **d_string_segments,
        int **d_numeric_segments
);
