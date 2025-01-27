#ifndef INDEXING_H
#define INDEXING_H

#include <thrust/device_vector.h>

// Function to count rows in a FastQ file
int count_rows_fastq(const thrust::device_vector<char> &d_data);

// Function to index newline characters in a FastQ file
void indexing_fields(
        const thrust::device_vector<char> &d_data,
        int row_count,
        thrust::device_vector<int> &fields_indices);

#endif // INDEXING_H
