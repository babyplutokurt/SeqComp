#include "rle.h"
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>

void rle_compress_int(
        const thrust::device_vector<int> &input,
        thrust::device_vector<int> &unique_keys,
        thrust::device_vector<int> &run_lengths,
        int &num_unique
)
{
    // Ensure output vectors have enough space initially
    unique_keys.resize(input.size());
    run_lengths.resize(input.size());

    auto end = thrust::reduce_by_key(
            input.begin(),                        // key begin
            input.end(),                          // key end
            thrust::make_constant_iterator(1),    // values (all 1)
            unique_keys.begin(),                  // output: unique keys
            run_lengths.begin(),                  // output: run lengths
            thrust::equal_to<int>(),              // key comparison
            thrust::plus<int>()                   // sum consecutive 1's
    );

    // The number of unique elements is how many "keys" we ended up with
    num_unique = static_cast<int>(end.first - unique_keys.begin());

    // Shrink output vectors to actual size
    unique_keys.resize(num_unique);
    run_lengths.resize(num_unique);
}

void rle_compress_char(
        const thrust::device_vector<char> &input,   // Input device vector
        thrust::device_vector<char> &unique_keys,  // Output unique keys
        thrust::device_vector<int> &run_lengths,   // Output run lengths
        int &num_unique                            // Output number of unique keys
) {
    // Resize the output vectors to match the input size
    unique_keys.resize(input.size());
    run_lengths.resize(input.size());

    // Perform reduce_by_key
    auto end = thrust::reduce_by_key(
            input.begin(),                         // Input keys
            input.end(),                           // End of input keys
            thrust::make_constant_iterator(1),     // Implicit values (all 1s)
            unique_keys.begin(),                   // Output unique keys
            run_lengths.begin(),                   // Output run lengths
            thrust::equal_to<char>(),              // Key comparison operator (default: equality)
            thrust::plus<int>()                    // Reduction operator (default: summation)
    );

    // Calculate the number of unique keys
    num_unique = end.first - unique_keys.begin();

    // Resize the output vectors to the actual size of the unique keys
    unique_keys.resize(num_unique);
    run_lengths.resize(num_unique);
}

void rle_decompress_char(
        const thrust::device_vector<char> &unique_keys,
        const thrust::device_vector<int> &run_lengths,
        int num_unique,
        thrust::device_vector<char> &output
) {
    // Compute total length
    int total_length = thrust::reduce(run_lengths.begin(), run_lengths.end(), 0);

    // Resize output
    output.resize(total_length);

    // Compute offsets using exclusive_scan
    thrust::device_vector<int> offsets(num_unique);
    thrust::exclusive_scan(run_lengths.begin(), run_lengths.end(), offsets.begin());

    // Now, for each index in [0, total_length), we need to find which run it belongs to.
    // We'll use upper_bound to map each index to a run index.
    // counting_iterator generates a sequence [0, 1, 2, ..., total_length-1]
    thrust::counting_iterator<int> cbegin(0);
    thrust::counting_iterator<int> cend(total_length);

    thrust::device_vector<int> run_indices(total_length);

    // upper_bound: For each element in [0, total_length), it finds the position
    // of the first offset greater than the element. We'll then subtract 1 to get the run index.
    thrust::upper_bound(offsets.begin(), offsets.end(), cbegin, cend, run_indices.begin());

    // Subtract 1 from each run index to get the correct run ID
    thrust::transform(run_indices.begin(), run_indices.end(),
                      thrust::make_constant_iterator(1),
                      run_indices.begin(), thrust::minus<int>());

    // Gather the corresponding keys for each position using run_indices
    thrust::gather(run_indices.begin(), run_indices.end(), unique_keys.begin(), output.begin());
}


/**
 * @brief Decompress an RLE "memory block" from the host, producing a device_vector<char>.
 *
 * The packed layout is:
 *  [int num_unique][char unique_keys[num_unique]][int run_lengths[num_unique]]
 *
 * @param compressed    [in] The RLE-compressed data on the host (std::vector<char>)
 * @param output        [out] Device vector where the decompressed result is stored
 */
void rle_decompress_memory(
        const std::vector<char> &compressed,   // single memory block in host
        thrust::device_vector<char> &output    // final decompressed data on device
)
{
    // 1) Read num_unique (4 bytes = sizeof(int))
    if (compressed.size() < sizeof(int)) {
        // Invalid or corrupted input
        output.clear();
        return;
    }
    int num_unique = *reinterpret_cast<const int*>(compressed.data());

    // 2) Next 'num_unique' bytes => unique keys
    size_t offset = sizeof(int); // we start reading after the int
    if (compressed.size() < offset + num_unique) {
        // Invalid input
        output.clear();
        return;
    }
    std::vector<char> host_keys(num_unique);
    std::memcpy(host_keys.data(), compressed.data() + offset, num_unique);
    offset += num_unique;

    // 3) Next 'num_unique' * sizeof(int) => run_lengths
    size_t run_lengths_bytes = num_unique * sizeof(int);
    if (compressed.size() < offset + run_lengths_bytes) {
        // Invalid input
        output.clear();
        return;
    }
    std::vector<int> host_runlengths(num_unique);
    std::memcpy(host_runlengths.data(),
                compressed.data() + offset,
                run_lengths_bytes);
    offset += run_lengths_bytes;

    // Now we have num_unique, host_keys[], host_runlengths[]

    // 4) Move these to device
    thrust::device_vector<char> dev_keys(host_keys.begin(), host_keys.end());
    thrust::device_vector<int>  dev_runlengths(host_runlengths.begin(), host_runlengths.end());

    // 5) Use rle_decompress to get final output
    rle_decompress_char(dev_keys, dev_runlengths, num_unique, output);
}