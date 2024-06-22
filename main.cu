#include <cuda.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <cstdint>

#include "sha256.cuh"

#define BLOCK_SIZE 256
#define GRID_SIZE 4096
#define ITEMS_PER_THREAD 1024

#define __both__ __host__ __device__

__device__ const char *baseline = "telaxion/zig+2070super+";
__device__ const int baseline_len = 23;

__both__ void memcpyHash(BYTE *dest, BYTE *src, int len) {
    for (int i = 0; i < len; i++) {
        dest[i] = src[i];
    }
}

__both__ int memcmpHash(BYTE *a, BYTE *b, int len) {
    for (int i = 0; i < len; i++) {
        if (a[i] < b[i]) {
            return -1;
        } else if (a[i] > b[i]) {
            return 1;
        }
    }
    return 0;
}

/// Creates a unique input string. Returns the length of the string.
__device__ int input_string(int epoch, int tid, int bid, int item,
                            BYTE *output) {
    uint elo = (epoch & 0x0F0F0F0F) + 0x61616161;
    uint ehi = ((epoch & 0xF0F0F0F0) >> 4) + 0x62626262;

    BYTE nonce[8];
    nonce[0] = (BYTE)('a' + ((bid >> 12) & 0xF));
    nonce[1] = (BYTE)('a' + ((bid >> 8) & 0xF));
    nonce[2] = (BYTE)('a' + ((bid >> 4) & 0xF));
    nonce[3] = (BYTE)('a' + (bid & 0xF));
    nonce[4] = (BYTE)('a' + ((tid >> 4) & 0xF));
    nonce[5] = (BYTE)('a' + (tid & 0xF));
    nonce[6] = (BYTE)('a' + ((item >> 4) & 0xF));
    nonce[7] = (BYTE)('a' + (item & 0xF));

    // concat baseline + ehi + elo + nonce
    int len = 0;
    for (int i = 0; i < baseline_len; i++) {
        output[len++] = baseline[i];
    }

    output[len++] = (char)((ehi >> 24) & 0xFF);
    output[len++] = (char)((ehi >> 16) & 0xFF);
    output[len++] = (char)((ehi >> 8) & 0xFF);
    output[len++] = (char)((ehi >> 0) & 0xFF);

    output[len++] = (char)((elo >> 24) & 0xFF);
    output[len++] = (char)((elo >> 16) & 0xFF);
    output[len++] = (char)((elo >> 8) & 0xFF);
    output[len++] = (char)((elo >> 0) & 0xFF);

    for (int i = 0; i < 8; i++) {
        output[len++] = nonce[i];
    }

    return len;
}

__both__ void print_hash(BYTE *hash) {
    for (int i = 0; i < 32; i++) {
        printf("%02x", hash[i]);
    }
}

__device__ BYTE global_smallestHash[SHA256_BLOCK_SIZE] = {0xFF};
__device__ BYTE global_smallestInput[128];

__both__ uint64_t byteswap64(uint64_t x) {
    return ((x & 0x00000000000000FFULL) << 56) |
           ((x & 0x000000000000FF00ULL) << 40) |
           ((x & 0x0000000000FF0000ULL) << 24) |
           ((x & 0x00000000FF000000ULL) << 8) |
           ((x & 0x000000FF00000000ULL) >> 8) |
           ((x & 0x0000FF0000000000ULL) >> 24) |
           ((x & 0x00FF000000000000ULL) >> 40) |
           ((x & 0xFF00000000000000ULL) >> 56);
}

__device__ volatile int sem = 0;

__device__ void acquire_semaphore(volatile int *lock) {
    while (atomicCAS((int *)lock, 0, 1) != 0)
        ;
}

__device__ void release_semaphore(volatile int *lock) {
    *lock = 0;
    __threadfence();
}

__global__ void setup() { memset(global_smallestHash, 0xFF, 32); }

__global__ void sha256_cuda(int epoch) {
    BYTE smallestHash[SHA256_BLOCK_SIZE];
    BYTE smallestInput[128];
    memset(smallestHash, 0xFF, 32);

    for (int seed = 0; seed < ITEMS_PER_THREAD; seed++) {
        BYTE input[128];
        int len = input_string(epoch, threadIdx.x, blockIdx.x, seed, input);

        SHA256_CTX ctx;
        sha256_init(&ctx);
        sha256_update(&ctx, (BYTE *)input, len);

        BYTE digest[SHA256_BLOCK_SIZE];
        sha256_final(&ctx, digest);

        int compare = memcmpHash(digest, smallestHash, 32);
        if (compare < 0) {
            memcpy(smallestHash, digest, 32);
            memcpy(smallestInput, input, 128);
        }
    }

    if (threadIdx.x == 0) {
        __syncthreads();
        acquire_semaphore(&sem);
    }

    bool smaller = memcmpHash(smallestHash, global_smallestHash, 32) < 0;
    if (smaller) {
        memcpy(global_smallestHash, smallestHash, SHA256_BLOCK_SIZE);
        memcpy(global_smallestInput, smallestInput, 128);
    }

    if (threadIdx.x == 0) {
        __threadfence();
        release_semaphore(&sem);
        __syncthreads();
    }
}

void printHashesPerSecond(long long hashes, double elapsedMillis) {
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

int main() {
    setup<<<1, 1>>>();

    // open out.txt for logging
    freopen("out.txt", "w", stdout);

    int epoch = 0;
    while (true) {
        auto start = std::chrono::high_resolution_clock::now();

        sha256_cuda<<<GRID_SIZE, BLOCK_SIZE>>>(epoch);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        double elapsedMillis =
            std::chrono::duration<double, std::milli>(end - start).count();

        BYTE resultHash[SHA256_BLOCK_SIZE];
        cudaMemcpyFromSymbol(resultHash, global_smallestHash,
                             sizeof(BYTE) * 32);
        print_hash(resultHash);

        char resultInput[128];
        cudaMemcpyFromSymbol(resultInput, global_smallestInput,
                             sizeof(BYTE) * 128);
        printf(" %s\n", resultInput);
        
        float hashes = (float)GRID_SIZE * BLOCK_SIZE * ITEMS_PER_THREAD;
        printHashesPerSecond(hashes, elapsedMillis);

        // save out.txt as it might crash next epoch
        fflush(stdout);

        epoch++;
    }

    return 0;
}
