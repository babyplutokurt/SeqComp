#pragma once

#include <string>
#include <vector>
#include <cstddef> // for size_t

// Holds the details of one parsed identifier
struct ParsedIdentifier {
    std::vector<std::string> segments;      // The actual segments
    std::vector<std::string> delimiters;    // Delimiters after each segment
    std::vector<int> segment_types;         // 0=string, 1=numeric
    std::vector<size_t> max_string_lengths; // length of each string-type segment
    std::vector<bool> segment_static;       // True=constant, False=variable

    int num_string_segments = 0;
    int num_numeric_segments = 0;
};

/**
 * @brief Parse a single identifier string into segments plus their delimiters.
 *
 * @param identifier       The entire identifier line from FASTQ (without newline).
 * @param delimiters_str   A string of delimiter characters (e.g. ". :/").
 * @return ParsedIdentifier
 */
ParsedIdentifier parse_identifier(const std::string &identifier,
                                  const std::string &delimiters_str = ". :/");

/**
 * @brief Sample up to 10 records from the device-based columnar arrays to determine
 *        which segments are constant vs. variable. The results are stored in
 *        pid.segment_static.
 *
 * @param pid             [in/out] The ParsedIdentifier; we fill pid.segment_static[i].
 * @param num_records     The total number of records on device.
 * @param d_string_arrays Array of device pointers for string segments.
 * @param d_numeric_arrays Array of device pointers for numeric segments.
 */
void sample_segments_on_device(
        ParsedIdentifier &pid,
        size_t num_records,
        const std::vector<char*> &d_string_arrays,
        const std::vector<int*> &d_numeric_arrays
);
