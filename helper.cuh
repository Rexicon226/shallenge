#include <stdio.h>

#include "sha256.cuh"

#define __both__ __host__ __device__

void printHashesPerSecond(long long hashes, double elapsedMillis);

inline void printHashesPerSecond(long long hashes, double elapsedMillis) {
    double hashesPerSecond = (hashes / (elapsedMillis / 1000.0));

    if (hashesPerSecond >= 1e12) {
        printf("%.2f TH/s\n", hashesPerSecond / 1e12);
    } else if (hashesPerSecond >= 1e9) {
        printf("%.2f GH/s\n", hashesPerSecond / 1e9);
    } else if (hashesPerSecond >= 1e6) {
        printf("%.2f MH/s\n", hashesPerSecond / 1e6);
    } else {
        printf("%.2f kH/s\n", hashesPerSecond / 1e3);
    }
}

#if (__CUDA_ARCH__)
__device__
#else
#endif
    const char valid_chars[] = "abcdefghijklmnopqrstuvwxyz0123456789";

#if (__CUDA_ARCH__)
__device__
#else
#endif
    const char *prefix = "telaxion/zig$2070super+";

const int num_valid_chars = sizeof(valid_chars) - 1;

__both__ int input_string(int epoch, int tid, int bid, int item, char *output) {
    int len = 0;

    memcpy(output, prefix, 23);
    len += 23;

    char *nonce_ptr = output + len;

    nonce_ptr[0] = valid_chars[(epoch >> 24) % num_valid_chars];
    nonce_ptr[1] = valid_chars[(epoch >> 16) % num_valid_chars];
    nonce_ptr[2] = valid_chars[(epoch >> 8) % num_valid_chars];
    nonce_ptr[3] = valid_chars[epoch % num_valid_chars];

    nonce_ptr[4] = valid_chars[(tid >> 16) % num_valid_chars];
    nonce_ptr[5] = valid_chars[(tid >> 8) % num_valid_chars];
    nonce_ptr[6] = valid_chars[tid % num_valid_chars];

    nonce_ptr[7] = valid_chars[(bid >> 16) % num_valid_chars];
    nonce_ptr[8] = valid_chars[(bid >> 8) % num_valid_chars];
    nonce_ptr[9] = valid_chars[bid % num_valid_chars];

    nonce_ptr[10] = valid_chars[(item >> 8) % num_valid_chars];
    nonce_ptr[11] = valid_chars[item % num_valid_chars];

    len += 12;

    return len;
}

__device__ void hashSeed(int epoch, int threadIdx, int blockIdx, int seed,
                         BYTE *digest) {
    char input[35];
    int len = input_string(epoch, threadIdx, blockIdx, seed, input);

    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_updateUnrolled(&ctx, (BYTE *)input);

    sha256_final(&ctx, digest);
}

__device__ uint64_t byteswap64(uint64_t x) {
    return ((x & 0x00000000000000FFULL) << 56) |
           ((x & 0x000000000000FF00ULL) << 40) |
           ((x & 0x0000000000FF0000ULL) << 24) |
           ((x & 0x00000000FF000000ULL) << 8) |
           ((x & 0x000000FF00000000ULL) >> 8) |
           ((x & 0x0000FF0000000000ULL) >> 24) |
           ((x & 0x00FF000000000000ULL) >> 40) |
           ((x & 0xFF00000000000000ULL) >> 56);
}