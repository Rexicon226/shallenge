#include <cuda.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <cstdint>

#include "helper.cuh"
#include "sha256.cuh"

#define BLOCK_SIZE 256
#define GRID_SIZE 1024
#define ITEMS_PER_THREAD 256

__device__ uint64_t s_totalHash[BLOCK_SIZE];

__device__ uint64_t s_inputString1[BLOCK_SIZE];
__device__ uint64_t s_inputString2[BLOCK_SIZE];

__device__ __forceinline__ void findHash(int epoch, int threadIdx, int blockIdx,
                                         int seed, uint64_t *smallestHash,
                                         uint64_t *smallestInput1,
                                         uint64_t *smallestInput2) {
    BYTE digest[SHA256_BLOCK_SIZE];
    hashSeed(epoch, threadIdx, blockIdx, seed, digest);

    uint64_t hash = *((uint64_t *)digest);
    hash = byteswap64(hash);

    if (hash < *smallestHash) {
        *smallestHash = hash;

        uint64_t packedInput = 0;
        packedInput = (packedInput << 32) | seed;
        packedInput = (packedInput << 32) | blockIdx;
        *smallestInput1 = packedInput;

        packedInput = 0;
        packedInput = (packedInput << 32) | threadIdx;
        packedInput = (packedInput << 32) | epoch;
        *smallestInput2 = packedInput;
    }
}

__global__ void setup() { memset(s_totalHash, 0xFF, sizeof(s_totalHash)); }

__global__ void kernel(int epoch, uint64_t *d_totalHash,
                       uint64_t *d_inputString1, uint64_t *d_inputString2) {
    __shared__ uint64_t s_smallestHash;

    __shared__ uint64_t s_smallestInput1;
    __shared__ uint64_t s_smallestInput2;

    bool thread_is_zero = threadIdx.x == 0;
    if (thread_is_zero) {
        s_smallestHash = 0xFFFFFFFFFFFFFFFF;
    }

    __syncthreads();

    uint64_t *smallestHash = &s_totalHash[threadIdx.x];

    uint64_t *smallestInput1 = &s_inputString1[threadIdx.x];
    uint64_t *smallestInput2 = &s_inputString2[threadIdx.x];

    for (int seed = 0; seed < ITEMS_PER_THREAD; seed++) {
        findHash(epoch, threadIdx.x, blockIdx.x, seed, smallestHash,
                 smallestInput1, smallestInput2);
    }

    if (thread_is_zero) {
        s_smallestHash = *smallestHash;
        s_smallestInput1 = *smallestInput1;
        s_smallestInput2 = *smallestInput2;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_totalHash[threadIdx.x + s] < s_totalHash[threadIdx.x]) {
                s_totalHash[threadIdx.x] = s_totalHash[threadIdx.x + s];
                s_inputString1[threadIdx.x] = s_inputString1[threadIdx.x + s];
                s_inputString2[threadIdx.x] = s_inputString2[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (thread_is_zero) {
        d_totalHash[blockIdx.x] = s_smallestHash;
        d_inputString1[blockIdx.x] = s_smallestInput1;
        d_inputString2[blockIdx.x] = s_smallestInput2;
    }
}

int main() {
    uint64_t *d_totalHash;

    uint64_t *d_inputString1;
    uint64_t *d_inputString2;

    cudaMalloc((void **)&d_totalHash, GRID_SIZE * sizeof(uint64_t));
    cudaMalloc((void **)&d_inputString1, GRID_SIZE * sizeof(uint64_t));
    cudaMalloc((void **)&d_inputString2, GRID_SIZE * sizeof(uint64_t));

    setup<<<1, 1>>>();

    int epoch = 0;
    while (true) {
        auto start = std::chrono::high_resolution_clock::now();

        kernel<<<GRID_SIZE, BLOCK_SIZE>>>(epoch, d_totalHash, d_inputString1,
                                          d_inputString2);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        double elapsedMillis =
            std::chrono::duration<double, std::milli>(end - start).count();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        uint64_t smallestHash;
        cudaMemcpy(&smallestHash, d_totalHash, sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);

        uint64_t smallestInputString1;
        uint64_t smallestInputString2;

        cudaMemcpy(&smallestInputString1, d_inputString1, sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&smallestInputString2, d_inputString2, sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);

        char input[35];

        int smallest_epoch = (int)(smallestInputString2 & 0xFFFFFFFF);
        int threadIdx = (int)((smallestInputString2 >> 32) & 0xFFFFFFFF);
        int blockIdx = (int)((smallestInputString1 & 0xFFFFFFFF));
        int seed = (int)((smallestInputString1 >> 32) & 0xFFFFFFFF);

        input_string(smallest_epoch, threadIdx, blockIdx, seed, input);

        printf("epoch %d: | ", smallest_epoch);
        for (int i = 0; i < 8; i++) {
            printf("%08x ", (uint32_t)(smallestHash >> (32 * (7 - i))));
        }
        printf("| %s\n", input);

        printf("elapsed time: %f\n", elapsedMillis);

        float hashes = (float)GRID_SIZE * BLOCK_SIZE * ITEMS_PER_THREAD;
        printHashesPerSecond(hashes, elapsedMillis);

        fflush(stdout);

        epoch++;
    }

    cudaFree(d_totalHash);
    cudaFree(d_inputString1);
    cudaFree(d_inputString2);

    return 0;
}