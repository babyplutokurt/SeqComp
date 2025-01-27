#include "rle.h"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

// Maximum chunk size per run segment when using uint8_t
static const int MAX_RUN = 255;

struct run_chunk_count_functor
{
    __host__ __device__
    int operator()(int length) const {
        // Number of chunks needed if each chunk is max 255 in length
        return (length + MAX_RUN - 1) / MAX_RUN;
    }
};

struct subtract_functor {
    __host__ __device__
    int operator()(int x, int y) const {
        return x - y;
    }
};

// Functor for assigning the final segmented runs after splitting
struct AssignSegmentFunctor
{
    int* run_indices;       // Maps final run index to original run index
    int* run_offsets;       // Prefix sums of segment counts
    int* temp_lengths;      // Original int run lengths from reduce_by_key
    char* temp_keys;        // Original unique keys from reduce_by_key
    char* unique_keys_out;  // Final unique keys array (uint8_t segments)
    uint8_t* run_lengths_out;
    int MAX_RUN;

    __host__ __device__
    void operator()(int i) const {
        int orig_run = run_indices[i];
        int orig_offset = run_offsets[orig_run]; // Where this run's segments start
        int segment_idx = i - orig_offset;        // segment index within this run

        int L = temp_lengths[orig_run];
        char K = temp_keys[orig_run];

        int segment_len = L - segment_idx * MAX_RUN;
        if (segment_len > MAX_RUN) {
            segment_len = MAX_RUN;
        }

        // Write the results
        unique_keys_out[i] = K;
        run_lengths_out[i] = static_cast<uint8_t>(segment_len);
    }
};

void rle_compress(
        const thrust::device_vector<char> &input,
        thrust::device_vector<char> &unique_keys_out,
        thrust::device_vector<uint8_t> &run_lengths_out,
        int &num_unique
) {
    // Temporary storage
    thrust::device_vector<char> temp_keys(input.size());
    thrust::device_vector<int> temp_lengths(input.size());

    // Initial reduce_by_key with int run-lengths
    auto end = thrust::reduce_by_key(
            input.begin(), input.end(),
            thrust::make_constant_iterator(1),
            temp_keys.begin(),
            temp_lengths.begin(),
            thrust::equal_to<char>(),
            thrust::plus<int>()
    );

    int raw_num_unique = static_cast<int>(end.first - temp_keys.begin());

    // Compute how many segments each run will be split into
    thrust::device_vector<int> run_segment_counts(raw_num_unique);
    thrust::transform(temp_lengths.begin(), temp_lengths.begin() + raw_num_unique,
                      run_segment_counts.begin(), run_chunk_count_functor());

    // Prefix sum to find total expanded run count
    thrust::device_vector<int> run_offsets(raw_num_unique);
    thrust::exclusive_scan(run_segment_counts.begin(), run_segment_counts.end(), run_offsets.begin());

    int expanded_count = run_offsets[raw_num_unique - 1] + run_segment_counts[raw_num_unique - 1];

    // Resize final arrays
    unique_keys_out.resize(expanded_count);
    run_lengths_out.resize(expanded_count);

    // Map final runs to original runs using upper_bound
    thrust::counting_iterator<int> final_run_ids(0);
    thrust::device_vector<int> run_indices(expanded_count);
    thrust::upper_bound(
            run_offsets.begin(), run_offsets.end(),
            final_run_ids, final_run_ids + expanded_count,
            run_indices.begin()
    );
    thrust::transform(run_indices.begin(), run_indices.end(),
                      thrust::make_constant_iterator(1),
                      run_indices.begin(),
                      subtract_functor()); // Now this matches the binary signature.


    // Get raw pointers for the functor
    int* d_run_indices = run_indices.data().get();
    int* d_run_offsets = run_offsets.data().get();
    int* d_temp_lengths = temp_lengths.data().get();
    char* d_temp_keys = temp_keys.data().get();
    char* d_unique_keys_out = unique_keys_out.data().get();
    uint8_t* d_run_lengths_out = run_lengths_out.data().get();

    // Assign segments in parallel
    thrust::for_each_n(
            thrust::device,
            thrust::counting_iterator<int>(0),
            expanded_count,
            AssignSegmentFunctor{
                    d_run_indices,
                    d_run_offsets,
                    d_temp_lengths,
                    d_temp_keys,
                    d_unique_keys_out,
                    d_run_lengths_out,
                    MAX_RUN
            }
    );

    num_unique = expanded_count;
}

void rle_decompress(
        const thrust::device_vector<char> &unique_keys,
        const thrust::device_vector<uint8_t> &run_lengths,
        int num_unique,
        thrust::device_vector<char> &output
) {
    // Compute total length
    int total_length = thrust::reduce(run_lengths.begin(), run_lengths.end(), 0, thrust::plus<int>());

    // Resize output
    output.resize(total_length);

    // Compute offsets using exclusive_scan
    thrust::device_vector<int> offsets(num_unique);
    thrust::exclusive_scan(run_lengths.begin(), run_lengths.end(), offsets.begin());

    thrust::counting_iterator<int> cbegin(0);
    thrust::counting_iterator<int> cend(total_length);
    thrust::device_vector<int> run_indices(total_length);

    thrust::upper_bound(
            offsets.begin(), offsets.end(),
            cbegin, cend,
            run_indices.begin()
    );

    thrust::transform(run_indices.begin(), run_indices.end(),
                      thrust::make_constant_iterator(1),
                      run_indices.begin(), subtract_functor());

    thrust::gather(run_indices.begin(), run_indices.end(), unique_keys.begin(), output.begin());
}
