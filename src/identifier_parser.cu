#include "identifier_parser.h"
#include <cctype>     // for std::isdigit
#include <cstring>    // for strcmp
#include <cuda_runtime.h> // for cudaMemcpy, etc.
#include <vector>

// Implementation of parse_identifier
ParsedIdentifier parse_identifier(const std::string &identifier,
                                  const std::string &delimiters_str)
{
    ParsedIdentifier pid;
    size_t pos = 0;
    while (pos < identifier.size()) {
        size_t dpos = identifier.find_first_of(delimiters_str, pos);
        if (dpos == std::string::npos) {
            dpos = identifier.size();
        }
        size_t seg_len = dpos - pos;

        if (seg_len > 0) {
            std::string seg = identifier.substr(pos, seg_len);
            pid.segments.push_back(seg);

            // Check if numeric
            bool all_digits = true;
            for (char c : seg) {
                if (!std::isdigit(static_cast<unsigned char>(c))) {
                    all_digits = false;
                    break;
                }
            }
            pid.segment_types.push_back(all_digits ? 1 : 0);

            // If not numeric, track its max length
            if (!all_digits) {
                pid.max_string_lengths.push_back(seg_len);
            }
        } else {
            // zero-length segment
            pid.segments.push_back("");
            pid.segment_types.push_back(0);
            pid.max_string_lengths.push_back(0);
        }

        // Delimiter
        if (dpos < identifier.size()) {
            pid.delimiters.push_back(std::string(1, identifier[dpos]));
        } else {
            pid.delimiters.push_back("");
        }

        pos = (dpos < identifier.size()) ? (dpos + 1) : dpos;
    }

    // Count how many are string vs numeric
    pid.num_string_segments = 0;
    pid.num_numeric_segments = 0;
    for (int t : pid.segment_types) {
        if (t == 0) pid.num_string_segments++;
        else        pid.num_numeric_segments++;
    }

    // Reserve space for static/variable flags
    pid.segment_static.resize(pid.segment_types.size(), true);

    return pid;
}

// Helper function: even sampling indices
static std::vector<size_t> build_sample_indices(size_t num_records, int max_samples = 10)
{
    std::vector<size_t> indices;
    if (num_records == 0) return indices;
    int actual_samples = (num_records < static_cast<size_t>(max_samples))
                         ? static_cast<int>(num_records)
                         : max_samples;
    indices.resize(actual_samples);

    for (int i = 0; i < actual_samples; ++i) {
        indices[i] = (num_records - 1) * i / (actual_samples - 1);
    }
    return indices;
}

// Implementation of sample_segments_on_device
void sample_segments_on_device(
        ParsedIdentifier &pid,
        size_t num_records,
        const std::vector<char*> &d_string_arrays,
        const std::vector<int*> &d_numeric_arrays
)
{
    // If we have <=1 record, trivially all segments are constant
    if (num_records <= 1) {
        pid.segment_static.assign(pid.segment_static.size(), true);
        return;
    }

    // Build up to 10 sample indices
    std::vector<size_t> sample_indices = build_sample_indices(num_records, 10);
    if (sample_indices.empty()) {
        // No records or some edge case
        return;
    }

    // Build segment->string array index or numeric array index
    std::vector<int> seg2str(pid.segment_types.size(), -1);
    std::vector<int> seg2num(pid.segment_types.size(), -1);

    {
        int s_count = 0, n_count = 0;
        for (size_t seg_idx = 0; seg_idx < pid.segment_types.size(); ++seg_idx) {
            if (pid.segment_types[seg_idx] == 0) {
                seg2str[seg_idx] = s_count++;
            } else {
                seg2num[seg_idx] = n_count++;
            }
        }
    }

    // For each segment, partial copy from device for each sample and compare
    for (size_t seg_idx = 0; seg_idx < pid.segment_types.size(); seg_idx++) {
        bool is_constant = true;

        if (pid.segment_types[seg_idx] == 0) {
            // A string segment
            int s_idx = seg2str[seg_idx];
            size_t max_len = pid.max_string_lengths[s_idx] + 1;

            // Copy first sample
            std::vector<char> first_val(max_len, 0);
            {
                size_t offset = sample_indices[0] * max_len;
                char *dev_ptr_offset = d_string_arrays[s_idx] + offset;
                cudaMemcpy(first_val.data(),
                           dev_ptr_offset,
                           max_len,
                           cudaMemcpyDeviceToHost);
            }

            // Compare with other samples
            for (size_t si = 1; si < sample_indices.size() && is_constant; ++si) {
                std::vector<char> sample_val(max_len, 0);
                size_t offset = sample_indices[si] * max_len;
                char *dev_ptr_offset = d_string_arrays[s_idx] + offset;
                cudaMemcpy(sample_val.data(),
                           dev_ptr_offset,
                           max_len,
                           cudaMemcpyDeviceToHost);

                if (std::strcmp(first_val.data(), sample_val.data()) != 0) {
                    is_constant = false;
                }
            }
        }
        else {
            // Numeric segment
            int n_idx = seg2num[seg_idx];

            // Copy first sample
            int first_val = 0;
            {
                size_t offset = sample_indices[0];
                int *dev_ptr_offset = d_numeric_arrays[n_idx] + offset;
                cudaMemcpy(&first_val,
                           dev_ptr_offset,
                           sizeof(int),
                           cudaMemcpyDeviceToHost);
            }

            // Compare
            for (size_t si = 1; si < sample_indices.size() && is_constant; ++si) {
                int sample_val = 0;
                size_t offset = sample_indices[si];
                int *dev_ptr_offset = d_numeric_arrays[n_idx] + offset;
                cudaMemcpy(&sample_val,
                           dev_ptr_offset,
                           sizeof(int),
                           cudaMemcpyDeviceToHost);

                if (sample_val != first_val) {
                    is_constant = false;
                }
            }
        }
        pid.segment_static[seg_idx] = is_constant;
    }
}
