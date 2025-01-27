#ifndef LORENZO_H
#define LORENZO_H

#include <thrust/device_vector.h>

// Function to compute an inclusive scan (prefix sum) on GPU
void inclusive_scan(const thrust::device_vector<int> &input, thrust::device_vector<int> &output);

// Function to compute Lorenzo 1D transformation on GPU
void lorenzo_1d(const thrust::device_vector<int> &input, thrust::device_vector<int> &output);

#endif // LORENZO_H