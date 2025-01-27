#include "gpulz.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////
// Macros and Constants
////////////////////////////////////////////////////////////////////////////////
#define BLOCK_SIZE   4096  // 4 KB block (in bytes)
#define THREAD_SIZE   128
#define WINDOW_SIZE   255
#define INPUT_TYPE    char   // For simplicity, compressing char data

// Error-checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err));                                \
            exit(-1);                                                       \
        }                                                                    \
    } while (0)

////////////////////////////////////////////////////////////////////////////////
// GPU Kernels
////////////////////////////////////////////////////////////////////////////////

// ------------------------------------------------------------------------
// compressKernelI: For each block of 4 KB, find LZ matches in parallel.
// Produces:
//   - lengthBuffer[], offsetBuffer[] for each position
//   - A local flag array (byteFlagArr[]), so we know which positions
//     are compressed tokens vs. literals
//   - A prefixBuffer[] to compute actual output offsets (intra-block).
// Finally writes partial results to:
//   - tmpFlagArrGlobal   (bit flags, block by block)
//   - tmpCompressedDataGlobal (raw compressed tokens, block by block)
// Also sets each block's compressedDataSizeGlobal & flagArrSizeGlobal
// ------------------------------------------------------------------------
template <typename T>
__global__ void compressKernelI(
        const T* __restrict__ d_in,
        uint32_t  numOfBlocks,
        uint32_t* flagArrSizeGlobal,
        uint32_t* compressedDataSizeGlobal,
        uint8_t*  tmpFlagArrGlobal,
        uint8_t*  tmpCompressedDataGlobal,
        int       minEncodeLength)
{
    // For T=char, blockSize = 4096
    const uint32_t blockSize = BLOCK_SIZE / sizeof(T);

    // Shared arrays
    __shared__ T      buffer[BLOCK_SIZE];           // loaded block
    __shared__ uint8_t lengthBuffer[BLOCK_SIZE];    // length of match
    __shared__ uint8_t offsetBuffer[BLOCK_SIZE];    // offset of match
    __shared__ uint32_t prefixBuffer[BLOCK_SIZE+1]; // prefix sums

    // We also track a local flag array (one bit per position -> 1/8 ratio).
    // 4096 bytes => max 4096/8 = 512 bytes of flags per block
    __shared__ uint8_t byteFlagArr[BLOCK_SIZE / 8];

    // 1) Load global->shared
    int tid  = threadIdx.x;
    int bIdx = blockIdx.x;
    for (int i = 0; i < blockSize/THREAD_SIZE; i++) {
        int idx = tid + i*THREAD_SIZE;
        buffer[idx] = d_in[bIdx*blockSize + idx];
    }
    __syncthreads();

    // 2) For each position in the block, find best match in [tid-WINDOW_SIZE, tid)
    for (int i = 0; i < blockSize/THREAD_SIZE; i++) {
        tid = threadIdx.x + i*THREAD_SIZE;

        int   windowStart = (tid - WINDOW_SIZE) < 0 ? 0 : (tid - WINDOW_SIZE);
        int   windowPtr   = windowStart;
        int   bufferStart = tid;
        int   bufferPtr   = bufferStart;

        uint8_t maxLen    = 0;
        uint8_t maxOffset = 0;
        uint8_t len       = 0;
        uint8_t offset    = 0;

        while (windowPtr < bufferStart && bufferPtr < (int)blockSize) {
            if (buffer[bufferPtr] == buffer[windowPtr]) {
                if (offset == 0) {
                    offset = bufferPtr - windowPtr;
                }
                len++;
                bufferPtr++;
            } else {
                if (len > maxLen) {
                    maxLen    = len;
                    maxOffset = offset;
                }
                len    = 0;
                offset = 0;
                bufferPtr = bufferStart;
            }
            windowPtr++;
        }
        // final check if we ended with the best
        if (len > maxLen) {
            maxLen    = len;
            maxOffset = offset;
        }

        lengthBuffer[tid] = maxLen;
        offsetBuffer[tid] = maxOffset;
        prefixBuffer[tid] = 0; // init for now
    }
    __syncthreads();

    // 3) Single-thread: build the local flag array + partial prefix sizes
    uint32_t flagCount = 0;
    if (threadIdx.x == 0) {
        uint8_t byteFlag      = 0;
        uint8_t flagPosition  = 0x01;

        int encodeIndex = 0;
        while (encodeIndex < (int)blockSize) {
            // If length < minEncodeLength => literal
            if (lengthBuffer[encodeIndex] < minEncodeLength) {
                prefixBuffer[encodeIndex] = sizeof(T);
                encodeIndex++;
            } else {
                // (length, offset)
                prefixBuffer[encodeIndex] = 2;
                // skip matched region in the lookahead
                encodeIndex += lengthBuffer[encodeIndex];
                byteFlag |= flagPosition;  // set bit
            }

            // Move to next bit
            if (flagPosition == 0x80) {
                byteFlagArr[flagCount] = byteFlag;
                flagCount++;
                flagPosition = 0x01;
                byteFlag     = 0;
            } else {
                flagPosition <<= 1;
            }
        }
        // store remainder if we didn't flush exactly
        if (flagPosition != 0x01) {
            byteFlagArr[flagCount] = byteFlag;
            flagCount++;
        }
    }
    __syncthreads();

    // 4) Prefix sum (upsweep-downsweep) in shared memory

    // Upsweep
    int stride = 1;
    for (int d = blockSize>>1; d > 0; d >>= 1) {
        for (int i = 0; i < blockSize/THREAD_SIZE; i++) {
            int t = threadIdx.x + i*THREAD_SIZE;
            if (t < d) {
                int ai = stride*(2*t+1) - 1;
                int bi = stride*(2*t+2) - 1;
                prefixBuffer[bi] += prefixBuffer[ai];
            }
        }
        __syncthreads();
        stride <<= 1;
    }

    // Clear last element, store total
    if (threadIdx.x == 0) {
        compressedDataSizeGlobal[bIdx] = prefixBuffer[blockSize-1];
        flagArrSizeGlobal[bIdx]        = flagCount;
        prefixBuffer[blockSize]        = prefixBuffer[blockSize-1];
        prefixBuffer[blockSize-1]      = 0;
    }
    __syncthreads();

    // Downsweep
    for (int d=1; d<=(int)blockSize; d<<=1) {
        stride >>= 1;
        for (int i = 0; i < blockSize/THREAD_SIZE; i++) {
            int t = threadIdx.x + i*THREAD_SIZE;
            if (t < d) {
                int ai = stride*(2*t+1) - 1;
                int bi = stride*(2*t+2) - 1;
                uint32_t tmp = prefixBuffer[ai];
                prefixBuffer[ai] = prefixBuffer[bi];
                prefixBuffer[bi] += tmp;
            }
        }
        __syncthreads();
    }

    // 5) Write actual compressed data into tmpCompressedDataGlobal
    //    using the prefix offsets
    int baseOffset = bIdx * blockSize * sizeof(T); // each block has up to 4096 bytes
    for (int i=0; i<blockSize/THREAD_SIZE; i++) {
        tid = threadIdx.x + i*THREAD_SIZE;
        uint32_t dstOffset = prefixBuffer[tid];
        if (prefixBuffer[tid+1] != dstOffset) {
            // If literal
            if (lengthBuffer[tid] < minEncodeLength) {
                // Copy one literal (sizeOf(T) = 1)
                tmpCompressedDataGlobal[baseOffset + dstOffset] = (uint8_t)buffer[tid];
            }
            else {
                // Copy 2 bytes: (length, offset)
                tmpCompressedDataGlobal[baseOffset + dstOffset  ] = lengthBuffer[tid];
                tmpCompressedDataGlobal[baseOffset + dstOffset+1] = offsetBuffer[tid];
            }
        }
    }

    // 6) Copy local flagArr to tmpFlagArrGlobal
    if (threadIdx.x == 0) {
        for (uint32_t i=0; i<flagCount; i++) {
            // each block has up to blockSize/8 = 512 bytes of flags
            tmpFlagArrGlobal[bIdx*(blockSize/8) + i] = byteFlagArr[i];
        }
    }
}

// ------------------------------------------------------------------------
// compressKernelIII: gathers all blocks' partial flags and partial compressed
// data into final contiguous arrays, using offset arrays generated by prefix sum
// ------------------------------------------------------------------------
template <typename T>
__global__ void compressKernelIII(
        uint32_t  numOfBlocks,
        const uint32_t* __restrict__ d_flagArrOffset,
        const uint32_t* __restrict__ d_compressedDataOffset,
        const uint8_t* __restrict__  tmpFlagArrGlobal,
        const uint8_t* __restrict__  tmpCompressedDataGlobal,
        uint8_t*                     d_flagArrGlobal,
        uint8_t*                     d_compressedDataGlobal)
{
    const uint32_t blockSize = BLOCK_SIZE / sizeof(T);
    int bIdx = blockIdx.x;
    int tid  = threadIdx.x;

    // Offsets where this block's data should land
    uint32_t flagOffset = d_flagArrOffset[bIdx];
    uint32_t flagSize   = d_flagArrOffset[bIdx+1] - flagOffset;

    uint32_t dataOffset = d_compressedDataOffset[bIdx];
    uint32_t dataSize   = d_compressedDataOffset[bIdx+1] - dataOffset;

    // Copy flags
    while (tid < (int)flagSize) {
        d_flagArrGlobal[flagOffset + tid] =
                tmpFlagArrGlobal[bIdx*(blockSize/8) + tid];
        tid += blockDim.x;
    }

    // Copy compressed data
    tid = threadIdx.x;
    while (tid < (int)dataSize) {
        d_compressedDataGlobal[dataOffset + tid] =
                tmpCompressedDataGlobal[bIdx*(blockSize)*sizeof(T) + tid];
        tid += blockDim.x;
    }
}

// ------------------------------------------------------------------------
// decompressKernel: Reconstruct one block from the final contiguous
// flags and compressed data arrays.
//   - Uses d_flagArrOffset / d_compressedDataOffset to find each block's region.
//   - The output (d_out) is the decompressed array (4 KB per block).
// ------------------------------------------------------------------------
template <typename T>
__global__ void decompressKernel(
        T*                    d_out,
        uint32_t             numOfBlocks,
        const uint32_t*      d_flagArrOffset,
        const uint32_t*      d_compressedDataOffset,
        const uint8_t*       d_flagArrGlobal,
        const uint8_t*       d_compressedDataGlobal)
{
    uint32_t bIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bIdx >= numOfBlocks) return;

    uint32_t blockSize = BLOCK_SIZE / sizeof(T);

    uint32_t flagStart  = d_flagArrOffset[bIdx];
    uint32_t flagEnd    = d_flagArrOffset[bIdx+1];
    uint32_t dataStart  = d_compressedDataOffset[bIdx];
    // dataEnd not strictly needed here

    // We'll reconstruct into d_out[bIdx * blockSize ... bIdx * blockSize + 4096)
    uint32_t dataPointsIndex = 0;
    uint32_t compressedIdx   = 0;

    for (uint32_t flagIdx = flagStart; flagIdx < flagEnd; flagIdx++) {
        uint8_t bits = d_flagArrGlobal[flagIdx];
        // Each byte has 8 bits => up to 8 tokens
        for (int bitCnt = 0; bitCnt < 8; bitCnt++) {
            int isMatch = (bits >> bitCnt) & 0x01;
            if (isMatch == 1) {
                // (length, offset) => 2 bytes
                uint8_t length = d_compressedDataGlobal[dataStart + compressedIdx];
                uint8_t offset = d_compressedDataGlobal[dataStart + compressedIdx + 1];
                compressedIdx += 2;

                uint32_t startPos = dataPointsIndex;
                for (int i=0; i<length; i++) {
                    d_out[bIdx*blockSize + dataPointsIndex] =
                            d_out[bIdx*blockSize + startPos - offset + i];
                    dataPointsIndex++;
                    if (dataPointsIndex >= blockSize) return;
                }
            }
            else {
                // literal => copy 1 byte (since T=char)
                d_out[bIdx*blockSize + dataPointsIndex] =
                        (T)d_compressedDataGlobal[dataStart + compressedIdx];
                compressedIdx += 1;
                dataPointsIndex++;
                if (dataPointsIndex >= blockSize) return;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Public API Implementation
//   gpulzCompress
//   gpulzDecompress
////////////////////////////////////////////////////////////////////////////////

extern "C"
void gpulzCompress(const char* d_input, uint32_t inputSize, GpuLZMeta* outMeta)
{
    // 1) Compute how many 4 KB blocks we need.
    // uint32_t fullBlocks = inputSize / BLOCK_SIZE;
    uint32_t remainder  = inputSize % BLOCK_SIZE;
    uint32_t padding    = (remainder == 0) ? 0 : (BLOCK_SIZE - remainder);
    uint32_t totalSize  = inputSize + padding;
    uint32_t numOfBlocks= totalSize / BLOCK_SIZE; // now multiple of 4096

    // Fill out some meta
    outMeta->numOfBlocks = numOfBlocks;
    outMeta->paddedSize  = padding;

    // 2) Allocate device arrays for per-block sizes:
    //    - compressedDataSizeGlobal[b], flagArrSizeGlobal[b]
    //    - prefix-sum offsets
    uint32_t* d_flagArrSizeGlobal         = nullptr;
    uint32_t* d_compressedDataSizeGlobal  = nullptr;
    uint32_t* d_flagArrOffsetGlobal       = nullptr;
    uint32_t* d_compressedDataOffsetGlobal= nullptr;

    CUDA_CHECK(cudaMalloc(&d_flagArrSizeGlobal,         sizeof(uint32_t)*(numOfBlocks+1)));
    CUDA_CHECK(cudaMalloc(&d_compressedDataSizeGlobal,  sizeof(uint32_t)*(numOfBlocks+1)));
    CUDA_CHECK(cudaMalloc(&d_flagArrOffsetGlobal,       sizeof(uint32_t)*(numOfBlocks+1)));
    CUDA_CHECK(cudaMalloc(&d_compressedDataOffsetGlobal,sizeof(uint32_t)*(numOfBlocks+1)));

    CUDA_CHECK(cudaMemset(d_flagArrSizeGlobal,          0, sizeof(uint32_t)*(numOfBlocks+1)));
    CUDA_CHECK(cudaMemset(d_compressedDataSizeGlobal,   0, sizeof(uint32_t)*(numOfBlocks+1)));
    CUDA_CHECK(cudaMemset(d_flagArrOffsetGlobal,        0, sizeof(uint32_t)*(numOfBlocks+1)));
    CUDA_CHECK(cudaMemset(d_compressedDataOffsetGlobal, 0, sizeof(uint32_t)*(numOfBlocks+1)));

    // 3) Temporary arrays for each block’s partial data
    //    - max 4096/8 = 512 bytes of flags per block
    //    - max 4096 bytes of compressed data per block
    uint32_t totalFlagsTmp  = numOfBlocks * (BLOCK_SIZE/8);
    uint32_t totalDataTmp   = numOfBlocks * BLOCK_SIZE; // T=char => same bytes
    uint8_t* d_tmpFlagArr   = nullptr;
    uint8_t* d_tmpCompData  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmpFlagArr,  totalFlagsTmp));
    CUDA_CHECK(cudaMalloc(&d_tmpCompData, totalDataTmp));

    // 4) Launch compressKernelI
    //    We’ll treat the input as `char*` => T=char
    const int BLOCKS = numOfBlocks;
    const int THREADS= THREAD_SIZE;
    int minEncodeLength = (sizeof(INPUT_TYPE) == 1) ? 2 : 1;

    compressKernelI<INPUT_TYPE><<<BLOCKS, THREADS>>>(
            d_input,
            numOfBlocks,
            d_flagArrSizeGlobal,
            d_compressedDataSizeGlobal,
            d_tmpFlagArr,
            d_tmpCompData,
            minEncodeLength
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5) Use CUB prefix sum to compute final offsets for flags/data
    {
        // Flag offsets
        void* d_tempStorage = nullptr;
        size_t tempStorageBytes = 0;
        cub::DeviceScan::ExclusiveSum(d_tempStorage, tempStorageBytes,
                                      d_flagArrSizeGlobal, d_flagArrOffsetGlobal,
                                      numOfBlocks+1);
        CUDA_CHECK(cudaMalloc(&d_tempStorage, tempStorageBytes));
        cub::DeviceScan::ExclusiveSum(d_tempStorage, tempStorageBytes,
                                      d_flagArrSizeGlobal, d_flagArrOffsetGlobal,
                                      numOfBlocks+1);
        CUDA_CHECK(cudaFree(d_tempStorage));
    }

    {
        // Data offsets
        void* d_tempStorage = nullptr;
        size_t tempStorageBytes = 0;
        cub::DeviceScan::ExclusiveSum(d_tempStorage, tempStorageBytes,
                                      d_compressedDataSizeGlobal, d_compressedDataOffsetGlobal,
                                      numOfBlocks+1);
        CUDA_CHECK(cudaMalloc(&d_tempStorage, tempStorageBytes));
        cub::DeviceScan::ExclusiveSum(d_tempStorage, tempStorageBytes,
                                      d_compressedDataSizeGlobal, d_compressedDataOffsetGlobal,
                                      numOfBlocks+1);
        CUDA_CHECK(cudaFree(d_tempStorage));
    }

    // 6) Read final total array sizes from last prefix elements
    uint32_t h_finalFlagSize       = 0;
    uint32_t h_finalCompressedSize = 0;
    CUDA_CHECK(cudaMemcpy(&h_finalFlagSize,
                          &d_flagArrOffsetGlobal[numOfBlocks],
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_finalCompressedSize,
                          &d_compressedDataOffsetGlobal[numOfBlocks],
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));

    outMeta->totalFlagArrSize         = h_finalFlagSize;
    outMeta->totalCompressedDataSize  = h_finalCompressedSize;

    // 7) Allocate final contiguous arrays for flags + compressed data
    CUDA_CHECK(cudaMalloc(&outMeta->d_flagArr,         h_finalFlagSize));
    CUDA_CHECK(cudaMalloc(&outMeta->d_compressedData,  h_finalCompressedSize));

    // 8) Launch compressKernelIII to gather scattered block data => contiguous
    compressKernelIII<INPUT_TYPE><<<BLOCKS, THREADS>>>(
            numOfBlocks,
            d_flagArrOffsetGlobal,
            d_compressedDataOffsetGlobal,
            d_tmpFlagArr,
            d_tmpCompData,
            outMeta->d_flagArr,
            outMeta->d_compressedData
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 9) Store offset arrays in outMeta (needed for decompress)
    outMeta->d_flagArrOffset        = d_flagArrOffsetGlobal;
    outMeta->d_compressedDataOffset = d_compressedDataOffsetGlobal;

    // We still need them to decompress block by block
    // Similarly, we keep d_flagArrSizeGlobal/d_compressedDataSizeGlobal only if needed.
    // (We don't strictly need them for the final decode, just their prefix sums.)

    // 10) Clean up (temporary arrays)
    // We keep d_flagArrSizeGlobal / d_compressedDataSizeGlobal only if you want them,
    // but here we can free them if you don't need them again.
    CUDA_CHECK(cudaFree(d_flagArrSizeGlobal));
    CUDA_CHECK(cudaFree(d_compressedDataSizeGlobal));

    CUDA_CHECK(cudaFree(d_tmpFlagArr));
    CUDA_CHECK(cudaFree(d_tmpCompData));

    // Done! The outMeta now fully describes the compressed layout.
}

// ---------------------------------------------------------------------------
extern "C"
void gpulzDecompress(const GpuLZMeta* inMeta, char* d_output, uint32_t outputSize)
{
    // We assume outputSize is the original data size (minus or plus any padding).
    // We'll just run a kernel that processes each block.
    uint32_t numOfBlocks = inMeta->numOfBlocks;

    // Decompression kernel config:
    // We'll have e.g. 32 threads (or 64, etc.) and enough blocks to cover all.
    dim3 blockDim(32);
    dim3 gridDim((numOfBlocks + blockDim.x - 1) / blockDim.x);

    // Launch
    decompressKernel<INPUT_TYPE><<<gridDim, blockDim>>>(
            (INPUT_TYPE*)d_output,
            numOfBlocks,
            inMeta->d_flagArrOffset,
            inMeta->d_compressedDataOffset,
            inMeta->d_flagArr,
            inMeta->d_compressedData
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
