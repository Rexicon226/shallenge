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


__both__ void print_hash(BYTE *hash) {
    for (int i = 0; i < 32; i++) {
        printf("%02x", hash[i]);
    }
}

template<int len>
__both__ inline int memcmpHash(BYTE *a, BYTE *b) {
    #pragma unroll len
    for (int i = 0; i < len; i++) {
        if (a[i] < b[i]) {
            return -1;
        } else if (a[i] > b[i]) {
            return 1;
        }
    }
    return 0;
}

template<int len> 
__device__ inline void memsetHash(int *dest, int x) {
    #pragma unroll len / sizeof(int)
    for (int i = 0; i < len / sizeof(int); i++) {
        dest[i] = x;
    }
}

template <int len>
__device__ inline void memcpyHash(BYTE *dest, BYTE *src) {
    #pragma unroll len
    for (int i = 0; i < len; i++) {
        dest[i] = src[i];
    }
}

__both__ int input_string(int epoch, int tid, int bid, int item,
                          char *output) {
    int len = 0;

    memcpy(output, "telaxion/zig+2070super+", 23);
    len += 23;

    char nonce[16];
    const char valid_chars[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    const int num_valid_chars = sizeof(valid_chars) - 1;

    nonce[0] = valid_chars[(epoch >> 24) % num_valid_chars];
    nonce[1] = valid_chars[(epoch >> 16) % num_valid_chars];
    nonce[2] = valid_chars[(epoch >> 8) % num_valid_chars];
    nonce[3] = valid_chars[epoch % num_valid_chars];

    nonce[4] = valid_chars[(tid >> 24) % num_valid_chars];
    nonce[5] = valid_chars[(tid >> 16) % num_valid_chars];
    nonce[6] = valid_chars[(tid >> 8) % num_valid_chars];
    nonce[7] = valid_chars[tid % num_valid_chars];

    nonce[8] = valid_chars[(bid >> 24) % num_valid_chars];
    nonce[9] = valid_chars[(bid >> 16) % num_valid_chars];
    nonce[10] = valid_chars[(bid >> 8) % num_valid_chars];
    nonce[11] = valid_chars[bid % num_valid_chars];

    nonce[12] = valid_chars[(item >> 24) % num_valid_chars];
    nonce[13] = valid_chars[(item >> 16) % num_valid_chars];
    nonce[14] = valid_chars[(item >> 8) % num_valid_chars];
    nonce[15] = valid_chars[item % num_valid_chars];

    memcpy(output + len, nonce, 16);
    len += 16;

    return len;
}

__device__ void hashSeed(int epoch, int threadIdx, int blockIdx, int seed, BYTE *digest) {
    char input[39];
    int len = input_string(epoch, threadIdx, blockIdx, seed, input);

    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (BYTE *)input, len);

    sha256_final(&ctx, digest);
}