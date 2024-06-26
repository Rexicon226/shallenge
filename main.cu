#include <cuda.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <cstdint>

#include "helper.cuh"
#include "sha256.cuh"

#define BLOCK_SIZE 256
#define GRID_SIZE 65536
#define ITEMS_PER_THREAD 256

#define GPU_COUNT 8

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

__global__ void setup() { memset(&s_totalHash, 0xFF, sizeof(s_totalHash)); }

__global__ void kernel(int device_id, int epoch, uint64_t *d_totalHash,
                       uint64_t *d_inputString1, uint64_t *d_inputString2) {
    __shared__ uint64_t s_smallestHash;

    __shared__ uint64_t s_smallestInput1;
    __shared__ uint64_t s_smallestInput2;

    if (threadIdx.x == 0) {
        memset(&s_smallestHash, 0xFF, sizeof(uint64_t));
    }

    __syncthreads();

    uint64_t *smallestHash = &s_totalHash[threadIdx.x];
    uint64_t *smallestInput1 = &s_inputString1[threadIdx.x];
    uint64_t *smallestInput2 = &s_inputString2[threadIdx.x];

    for (int seed = 0; seed < ITEMS_PER_THREAD; seed++) {
        findHash(epoch, threadIdx.x, blockIdx.x, seed, smallestHash,
                 smallestInput1, smallestInput2);
    }

    if (threadIdx.x == 0) {
        s_smallestHash = *smallestHash;
        s_smallestInput1 = *smallestInput1;
        s_smallestInput2 = *smallestInput2;
    }

    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_totalHash[threadIdx.x + s] < s_totalHash[threadIdx.x]) {
                s_totalHash[threadIdx.x] = s_totalHash[threadIdx.x + s];
                s_inputString1[threadIdx.x] = s_inputString1[threadIdx.x + s];
                s_inputString2[threadIdx.x] = s_inputString2[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_totalHash[blockIdx.x] = s_smallestHash;
        d_inputString1[blockIdx.x] = s_smallestInput1;
        d_inputString2[blockIdx.x] = s_smallestInput2;
    }
}

int main() {
    printf("Number of GPUs: %d\n", GPU_COUNT);

    uint64_t *d_totalHash[GPU_COUNT];

    uint64_t *d_inputString1[GPU_COUNT];
    uint64_t *d_inputString2[GPU_COUNT];

    uint64_t current_totalHash;
    uint64_t current_inputString1;
    uint64_t current_inputString2;

    memset(&current_totalHash, 0xFF, sizeof(uint64_t));

    for (unsigned int device_id = 0; device_id < GPU_COUNT; device_id++) {
        cudaSetDevice(device_id);
        cudaMalloc((void **)&d_totalHash[device_id],
                   GRID_SIZE * sizeof(uint64_t));
        cudaMalloc((void **)&d_inputString1[device_id],
                   GRID_SIZE * sizeof(uint64_t));
        cudaMalloc((void **)&d_inputString2[device_id],
                   GRID_SIZE * sizeof(uint64_t));
        setup<<<1, 1>>>();
    }

    int epoch = 0;
    while (true) {
        auto start = std::chrono::high_resolution_clock::now();

        for (unsigned int device_id = 0; device_id < GPU_COUNT; device_id++) {
            cudaSetDevice(device_id);
            kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
                device_id, epoch, d_totalHash[device_id],
                d_inputString1[device_id], d_inputString2[device_id]);
            epoch++;
        }

        for (unsigned int device_id = 0; device_id < GPU_COUNT; device_id++) {
            cudaSetDevice(device_id);
            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsedMillis =
            std::chrono::duration<double, std::milli>(end - start).count();

        float hashes =
            (float)GRID_SIZE * BLOCK_SIZE * ITEMS_PER_THREAD * GPU_COUNT;

        printHashesPerSecond(hashes, elapsedMillis);

        // find the global best hash, and update all d_totalHash pointers to be
        // that in order to synchronize the gpus

        for (unsigned int device_id = 0; device_id < GPU_COUNT; device_id++) {
            uint64_t smallestHash;
            cudaMemcpy(&smallestHash, d_totalHash[device_id], sizeof(uint64_t),
                       cudaMemcpyDeviceToHost);

            uint64_t smallestInputString1;
            uint64_t smallestInputString2;

            cudaMemcpy(&smallestInputString1, d_inputString1[device_id],
                       sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&smallestInputString2, d_inputString2[device_id],
                       sizeof(uint64_t), cudaMemcpyDeviceToHost);

            if (current_totalHash > smallestHash) {
                current_totalHash = smallestHash;
                current_inputString1 = smallestInputString1;
                current_inputString2 = smallestInputString2;
            }
        }

        // update all d_totalHashes for the current best hash
        for (unsigned int device_id = 0; device_id < GPU_COUNT; device_id++) {
            cudaMemcpy(d_totalHash[device_id], &current_totalHash,
                       sizeof(uint64_t), cudaMemcpyHostToDevice);
        }

        char input[128];
        int smallest_epoch = (int)(current_inputString2 & 0xFFFFFFFF);
        int threadIdx = (int)((current_inputString2 >> 32) & 0xFFFFFFFF);
        int blockIdx = (int)((current_inputString1 & 0xFFFFFFFF));
        int seed = (int)((current_inputString1 >> 32) & 0xFFFFFFFF);

        input_string(smallest_epoch, threadIdx, blockIdx, seed, input);

        printf("epoch %d: | ", epoch);
        printf("| %016lx | %s\n", current_totalHash, input);

        if (current_totalHash == 0) break;

        printf("elapsed time: %f\n", elapsedMillis);
        fflush(stdout);
    }

    cudaFree(d_totalHash);
    cudaFree(d_inputString1);
    cudaFree(d_inputString2);

    return 0;
}
