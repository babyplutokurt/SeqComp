#ifndef GPULZ_H
#define GPULZ_H

#include <cstdint>

// ---------------------------------------------------------------------------
// A small metadata struct describing the compressed layout on GPU
// ---------------------------------------------------------------------------
typedef struct {
    // Final contiguous arrays of flags and compressed bytes
    uint8_t* d_flagArr;
    uint8_t* d_compressedData;

    // Per-block offsets (in device memory) so decompression can find each block's data
    uint32_t* d_flagArrOffset;
    uint32_t* d_compressedDataOffset;

    // Number of 4 KB blocks used to compress the input
    uint32_t  numOfBlocks;

    // Final total sizes (in bytes) of the arrays above
    uint32_t  totalFlagArrSize;
    uint32_t  totalCompressedDataSize;

    // How many bytes of padding we used (if inputSize wasn't multiple of BLOCK_SIZE)
    uint32_t  paddedSize;
} GpuLZMeta;

// ---------------------------------------------------------------------------
// Exported C-style functions to compress or decompress
//   gpulzCompress(...)   -> compress from a device pointer
//   gpulzDecompress(...) -> decompress into a device pointer
// ---------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

// Compress a device buffer of size `inputSize` bytes.
// On output, fills out the GpuLZMeta struct with device pointers
// (d_flagArr, d_compressedData, offsets, etc.) and sizes.
// The caller is responsible for eventually freeing the memory
// from outMeta’s device pointers (via cudaFree).
void gpulzCompress(const char* d_input,       // [device pointer]
                   uint32_t     inputSize,    // size in bytes
                   GpuLZMeta*   outMeta);     // output struct with metadata

// Decompress data using the compressed arrays described in `inMeta`.
// Writes the result into d_output, which must be allocated with at least
// `outputSize` bytes on the device. Typically, outputSize == inputSize.
void gpulzDecompress(const GpuLZMeta* inMeta,
                     char*            d_output,
                     uint32_t         outputSize);

#ifdef __cplusplus
}
#endif

#endif // GPULZ_H
