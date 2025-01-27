#include "identifier_compressor.h"
#include "lorenzo.h"     // for lorenzo_1d(...)
#include "rle.h"         // for rle_compress_int(...)
#include "gpulz.h"       // for gpulzCompress(...)
#include <thrust/device_vector.h>
#include <cstring>       // for memcpy, etc.

CompressedIdentifier compress_identifier(
        const ParsedIdentifier &pid,
        size_t num_records,
        const std::vector<char*> &d_string_arrays,
        const std::vector<int*> &d_numeric_arrays
)
{
    CompressedIdentifier cident;
    cident.segments.resize(pid.segments.size());

    // segment -> string index or numeric index
    std::vector<int> seg2str(pid.segments.size(), -1);
    std::vector<int> seg2num(pid.segments.size(), -1);
    {
        int s_count = 0, n_count = 0;
        for (size_t i = 0; i < pid.segments.size(); i++) {
            if (pid.segment_types[i] == 0) {
                seg2str[i] = s_count++;
            } else {
                seg2num[i] = n_count++;
            }
        }
    }

    for (size_t seg_idx = 0; seg_idx < pid.segments.size(); seg_idx++) {
        CompressedSegment &cseg = cident.segments[seg_idx];
        cseg.segment_type = pid.segment_types[seg_idx];
        cseg.is_static    = pid.segment_static[seg_idx];
        cseg.delimiter    = pid.delimiters[seg_idx];
        cseg.is_compressed = false;
        cseg.is_gpulz     = false; // default

        // 1) STATIC SEGMENT
        if (cseg.is_static) {
            if (cseg.segment_type == 0) {
                // Static STRING => store just the 1st record
                int s_idx = seg2str[seg_idx];
                size_t max_len = pid.max_string_lengths[s_idx] + 1;
                std::vector<char> hostBuf(max_len, 0);

                cudaMemcpy(hostBuf.data(),
                           d_string_arrays[s_idx],
                           max_len,
                           cudaMemcpyDeviceToHost);

                cseg.static_string_value = hostBuf.data();
                cseg.compressed_size     = cseg.static_string_value.size();
            }
            else {
                // Static NUMERIC => store the int from record 0
                int n_idx = seg2num[seg_idx];
                int val = 0;
                cudaMemcpy(&val,
                           d_numeric_arrays[n_idx],
                           sizeof(int),
                           cudaMemcpyDeviceToHost);

                cseg.static_numeric_value = val;
                cseg.compressed_size      = sizeof(int);
            }
        }

            // 2) VARIABLE SEGMENT
        else {
            if (cseg.segment_type == 0) {
                // VARIABLE STRING => use GPULZ
                // a) Determine how big the entire column is
                int s_idx = seg2str[seg_idx];
                size_t max_len = pid.max_string_lengths[s_idx] + 1;
                size_t total_bytes = num_records * max_len;

                // b) Call gpulzCompress(...) to compress on GPU
                gpulzCompress(d_string_arrays[s_idx],    // device pointer
                              static_cast<uint32_t>(total_bytes),
                              &cseg.gpuLZData);      // store metadata in cseg

                // c) Mark we used GPULZ
                cseg.is_compressed = true;
                cseg.is_gpulz      = true;
                // The final size is the sum of the two arrays (flags + data)
                cseg.compressed_size = cseg.gpuLZData.totalFlagArrSize
                                       + cseg.gpuLZData.totalCompressedDataSize;
            }
            else {
                // VARIABLE NUMERIC => Lorenzo + RLE
                int n_idx = seg2num[seg_idx];
                thrust::device_vector<int> d_full(num_records);
                cudaMemcpy(thrust::raw_pointer_cast(d_full.data()),
                           d_numeric_arrays[n_idx],
                           num_records * sizeof(int),
                           cudaMemcpyDeviceToDevice);

                // b) Lorenzo
                thrust::device_vector<int> d_lorenzoOut(num_records);
                lorenzo_1d(d_full, d_lorenzoOut);

                // c) RLE
                thrust::device_vector<int> d_uniqueInts;
                thrust::device_vector<int> d_runLengths;
                int num_unique = 0;
                rle_compress_int(d_lorenzoOut, d_uniqueInts, d_runLengths, num_unique);

                size_t compressed_bytes = sizeof(int)
                                          + num_unique * sizeof(int)
                                          + num_unique * sizeof(int);
                size_t original_bytes = num_records * sizeof(int);

                if (compressed_bytes > original_bytes) {
                    // store original
                    cseg.is_compressed = false;
                    cseg.numeric_compressed_data.resize(original_bytes);
                    cudaMemcpy(cseg.numeric_compressed_data.data(),
                               thrust::raw_pointer_cast(d_full.data()),
                               original_bytes,
                               cudaMemcpyDeviceToHost);
                    cseg.compressed_size = original_bytes;
                } else {
                    // keep compressed
                    cseg.numeric_compressed_data.resize(compressed_bytes);
                    char* base_ptr = cseg.numeric_compressed_data.data();
                    std::memcpy(base_ptr, &num_unique, sizeof(int));
                    size_t offset = sizeof(int);

                    // unique keys
                    if (num_unique > 0) {
                        cudaMemcpy(base_ptr + offset,
                                   thrust::raw_pointer_cast(d_uniqueInts.data()),
                                   num_unique * sizeof(int),
                                   cudaMemcpyDeviceToHost);
                    }
                    offset += num_unique * sizeof(int);

                    // run lengths
                    if (num_unique > 0) {
                        cudaMemcpy(base_ptr + offset,
                                   thrust::raw_pointer_cast(d_runLengths.data()),
                                   num_unique * sizeof(int),
                                   cudaMemcpyDeviceToHost);
                    }
                    offset += num_unique * sizeof(int);

                    cseg.is_compressed   = true;
                    cseg.compressed_size = compressed_bytes;
                }
            }
        }
    } // end for seg_idx

    return cident;
}
