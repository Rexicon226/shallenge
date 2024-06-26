#include <stdio.h>

#include <cmath>

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
    const char valid_chars[] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

#if (__CUDA_ARCH__)
__device__
#else
#endif
    const char *prefix = "telaxion/zig+dgxa100+";

const int num_valid_chars = sizeof(valid_chars) - 1;

__both__ int input_string(int epoch, int tid, int bid, int item, char *output) {
    int len = 0;

    memcpy(output, prefix, 21);
    len += 21;

    while (epoch > 0) {
        int remainder = epoch % num_valid_chars;
        output[len++] = valid_chars[remainder];
        epoch /= num_valid_chars;
    }

    while (tid > 0) {
        int remainder = tid % num_valid_chars;
        output[len++] = valid_chars[remainder];
        tid /= num_valid_chars;
    }

    while (bid > 0) {
        int remainder = bid % num_valid_chars;
        output[len++] = valid_chars[remainder];
        bid /= num_valid_chars;
    }

    while (item > 0) {
        int remainder = item % num_valid_chars;
        output[len++] = valid_chars[remainder];
        item /= num_valid_chars;
    }

    output[len] = '\0';

    return len;
}

__device__ void hashSeed(int epoch, int threadIdx, int blockIdx, int seed,
                         BYTE *digest) {
    char input[128];
    int len = input_string(epoch, threadIdx, blockIdx, seed, input);

    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (BYTE *)input, len);

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