#include "basecall_compress.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(-1);                                             \
        }                                                         \
    } while(0)

/******************************************************************************
 * 1) calcBasecallLengthsFromFields
 *    basecall_length[n] = (fields_indices[4n+2] - 1) - fields_indices[4n+1]
 ******************************************************************************/
__global__ void calcBasecallLineLengthsKernel(
        const int *d_fields_indices,
        int *d_basecall_length,
        int num_records
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_records) return;

    // basecall line => 4*idx + 1
    int base_idx = 4*idx + 1;
    int start = d_fields_indices[base_idx];
    int end   = d_fields_indices[base_idx + 1] - 1; // skip newline
    int length = end - start;
    d_basecall_length[idx] = (length < 0) ? 0 : length;
}

void calcBasecallLengthsFromFields(
        const int *d_fields_indices,
        int *d_basecall_length,
        int num_records
)
{
    int blockSize = 256;
    int gridSize = (num_records + blockSize - 1)/blockSize;

    calcBasecallLineLengthsKernel<<<gridSize, blockSize>>>(
            d_fields_indices,
            d_basecall_length,
            num_records
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/******************************************************************************
 * 2) calcBasecallOffsets
 *    prefix sum of basecall_length => offset array
 *    offset[i] = sum of length[0..i-1], offset[0]=0
 *    offset[num_records] = total bases
 ******************************************************************************/
void calcBasecallOffsets(
        const int *d_basecall_length,
        int num_records,
        int *d_offset
)
{
    // We'll do an inclusive scan approach => offset[0]=0, offset[i+1] = offset[i] + length[i]
    // final array size is num_records+1
    // For convenience, we create a thrust vector from d_basecall_length
    thrust::device_vector<int> d_len(num_records);
    thrust::copy_n(d_basecall_length, num_records, d_len.begin());

    // We also create a device_vector<int> of size num_records+1 for offsets
    thrust::device_vector<int> d_offsetDV(num_records+1);

    d_offsetDV[0] = 0;
    // inclusive scan => store in d_offsetDV[1..num_records]
    thrust::inclusive_scan(d_len.begin(), d_len.end(), d_offsetDV.begin() + 1);

    // now copy back to d_offset (raw pointer)
    thrust::copy_n(d_offsetDV.begin(), num_records+1, d_offset);
}

/******************************************************************************
 * 3) basecallCompress => calls above steps + compress kernel
 ******************************************************************************/
__device__ inline uint8_t baseTo2bits(char b)
{
    switch(b) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default:  return 0; // 'N' => 0
    }
}

// We'll do 1 thread per record
__global__ void compressBasecallKernel(
        const char *d_data,
        const int  *d_fields_indices,
        const int  *d_offset,           // prefix sum (num_records+1)
        const int  *d_basecall_length,
        int         num_records,
        uint8_t    *d_compressed,
        int        *d_nPositions,    // store absolute indices of 'N'
        int        *d_nCounter
)
{
    int rec_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (rec_id >= num_records) return;

    // basecall line => 4*rec_id + 1
    int basecall_lineIdx = 4*rec_id + 1;
    int start_line = d_fields_indices[basecall_lineIdx];

    // basecall line length
    int length = d_basecall_length[rec_id];

    // bit offset = 2 * d_offset[rec_id]
    // because each char => 2 bits
    size_t bitOffset = (size_t)d_offset[rec_id] * 2;

    for (int i=0; i<length; i++) {
        char base = d_data[start_line + i];

        // If 'N', store absolute index => (start_line + i)
        if (base == 'N') {
            int idx = atomicAdd(d_nCounter, 1);
            // Store position in decompressed array: d_offset[rec_id] + i
            d_nPositions[idx] = d_offset[rec_id] + i;  // Corrected line
            base = 'A'; // store 'A' => bits=0
        }


        uint8_t code = baseTo2bits(base);
        size_t myBit = bitOffset + (size_t)(i*2);

        size_t byteIdx = myBit >> 3;
        int    bitInByte = myBit & 0x7;

        uint8_t oldVal = d_compressed[byteIdx];
        uint8_t shifted = (code & 0x3) << bitInByte;
        uint8_t mask    = 0x3 << bitInByte;
        oldVal &= ~mask;
        oldVal |= shifted;
        d_compressed[byteIdx] = oldVal;
    }
}

void basecallCompress(
        const char   *d_data,
        const int    *d_fields_indices,
        int           row_count,
        int           num_records,
        BasecallCompressMeta &outMeta
)
{
    // 1) compute basecall_length array
    int *d_basecall_length = nullptr;
    CUDA_CHECK(cudaMalloc(&d_basecall_length, num_records*sizeof(int)));
    calcBasecallLengthsFromFields(d_fields_indices, d_basecall_length, num_records);

    // 2) compute prefix sum => d_offset
    int *d_offset = nullptr;
    CUDA_CHECK(cudaMalloc(&d_offset, (num_records+1)*sizeof(int)));
    calcBasecallOffsets(d_basecall_length, num_records, d_offset);

    // 3) read total_chars => d_offset[num_records]
    int total_chars = 0;
    CUDA_CHECK(cudaMemcpy(&total_chars, d_offset+num_records, sizeof(int), cudaMemcpyDeviceToHost));

    size_t total_bits  = (size_t)total_chars * 2;
    size_t total_bytes = (total_bits + 7)/8;

    outMeta.total_bits  = total_bits;
    outMeta.total_bytes = total_bytes;

    // allocate compressed array
    CUDA_CHECK(cudaMalloc(&outMeta.d_compressed, total_bytes));
    CUDA_CHECK(cudaMemset(outMeta.d_compressed, 0, total_bytes));

    // N array => worst-case all are 'N' => total_chars
    // This time, we store 1 int per 'N' => absolute index
    CUDA_CHECK(cudaMalloc(&outMeta.d_nPositions, (size_t)total_chars*sizeof(int)));
    outMeta.n_count = 0;

    int *d_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    // kernel => 1 thread/record
    int blockSize=256;
    int gridSize=(num_records + blockSize -1)/blockSize;
    compressBasecallKernel<<<gridSize, blockSize>>>(
            d_data,
            d_fields_indices,
            d_offset,
            d_basecall_length,
            num_records,
            outMeta.d_compressed,
            outMeta.d_nPositions,
            d_counter
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // read n_count
    int h_nCount=0;
    CUDA_CHECK(cudaMemcpy(&h_nCount, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    outMeta.n_count = h_nCount;

    // cleanup
    CUDA_CHECK(cudaFree(d_counter));
    CUDA_CHECK(cudaFree(d_offset));
    CUDA_CHECK(cudaFree(d_basecall_length));
}

/******************************************************************************
 * 4) basecallDecompress => chunk-based + restore 'N'
 ******************************************************************************/
__device__ inline char bitsToBase(uint8_t code)
{
    switch(code) {
        case 0: return 'A';
        case 1: return 'C';
        case 2: return 'G';
        case 3: return 'T';
        default:return 'A';
    }
}

__global__ void decompressChunksKernel(
        const uint8_t *d_inBits,
        size_t total_bytes,
        char* d_out,
        int chunkSizeBytes,
        size_t total_bases
)
{
    int chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t start_byte = (size_t)chunk_idx * chunkSizeBytes;
    if (start_byte >= total_bytes) return;

    size_t max_bytes = (start_byte + chunkSizeBytes <= total_bytes)
                       ? chunkSizeBytes
                       : (total_bytes - start_byte);
    size_t outBaseIdx = (size_t)chunk_idx * (chunkSizeBytes * 4);

    size_t outCount = max_bytes * 4;
    if (outBaseIdx + outCount > total_bases) {
        outCount = (outBaseIdx + outCount > total_bases)
                   ? (total_bases - outBaseIdx)
                   : outCount;
    }

    for (size_t i=0; i<max_bytes; i++) {
        uint8_t val = d_inBits[start_byte + i];
        for (int b=0; b<4; b++) {
            size_t outPos = outBaseIdx + i*4 + b;
            if (outPos >= total_bases) return;
            uint8_t code = (val >> (b*2)) & 0x3;
            d_out[outPos] = bitsToBase(code);
        }
    }
}

// --- Updated restoreNKernel with bounds checking ---
__global__ void restoreNKernel(
        const int *d_nPositions,
        int n_count,
        char *d_decomp,
        int total_basecalls  // Added parameter for bounds check
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_count) return;

    int absIdx = d_nPositions[idx];
    // Ensure the index is within valid range before writing
    if (absIdx >= 0 && absIdx < total_basecalls) {
        d_decomp[absIdx] = 'N';
    }
}
void basecallDecompress(
        const BasecallCompressMeta &outMeta,
        char *d_output,
        size_t total_basecalls,
        int chunkSizeBytes,
        const int *d_fields_indices  // we no longer need this for 'N' restoration
        // if we do absolute indexes
)
{
    // chunk-based
    int blockSize = 128;
    size_t chunks = (outMeta.total_bytes + chunkSizeBytes -1)/chunkSizeBytes;
    int gridSize = (int)((chunks + blockSize -1)/blockSize);

    decompressChunksKernel<<<gridSize, blockSize>>>(
            outMeta.d_compressed,
            outMeta.total_bytes,
            d_output,
            chunkSizeBytes,
            total_basecalls
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Updated kernel launch in basecallDecompress ---
    // restore 'N'
    if (outMeta.n_count > 0 && outMeta.d_nPositions != nullptr) {
        int nCount = outMeta.n_count;
        blockSize = 256;
        gridSize = (nCount + blockSize -1)/blockSize;

        restoreNKernel<<<gridSize, blockSize>>>(
                outMeta.d_nPositions,
                nCount,
                d_output,
                total_basecalls  // Pass total_basecalls for bounds check
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
