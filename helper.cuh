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