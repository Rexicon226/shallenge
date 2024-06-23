#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <chrono>

#include "sha256.cuh"
#include "helper.cuh"

#define BLOCK_SIZE 1024
#define GRID_SIZE 5120
#define ITEMS_PER_THREAD 32

__device__ BYTE s_totalHash[BLOCK_SIZE][SHA256_BLOCK_SIZE];
__device__ int s_inputString[BLOCK_SIZE][4];

__device__ inline void findHash(int epoch, int threadIdx, int blockIdx, int seed, BYTE *smallestHash, int *smallestInput) {
    BYTE digest[39];
    hashSeed(epoch, threadIdx, blockIdx, seed, digest);

    int compare = memcmpHash<32>(digest, smallestHash);
    if (compare < 0) {
        memcpy(smallestHash, digest, SHA256_BLOCK_SIZE);
        smallestInput[0] = threadIdx;
        smallestInput[1] = blockIdx;
        smallestInput[2] = seed;
        smallestInput[3] = epoch;
    }
}

__global__ void setup() {
    memset(s_totalHash, 0xFF, sizeof(s_totalHash));
}

__global__ void kernel(int epoch, BYTE *d_totalHash, int *d_inputString) {
    __shared__ BYTE s_smallestHash[SHA256_BLOCK_SIZE];
    __shared__ int s_smallestInput[3];

    if (threadIdx.x == 0) {
        memset(s_smallestHash, 0xFF, sizeof(s_smallestHash));
    }

    __syncthreads();

    BYTE *smallestHash = s_totalHash[threadIdx.x];
    int *smallestInput = s_inputString[threadIdx.x];
    
    for (int seed = 0; seed < ITEMS_PER_THREAD; seed++) {
        findHash(epoch, threadIdx.x, blockIdx.x, seed, smallestHash, smallestInput);
    }

    if (threadIdx.x == 0) {
        memcpy(s_smallestHash, smallestHash, SHA256_BLOCK_SIZE);
        memcpy(s_smallestInput, smallestInput, 4 * sizeof(int));
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            int compare = memcmpHash<32>(s_totalHash[threadIdx.x], s_totalHash[threadIdx.x + s]);
            if (compare > 0) {
                memcpy(s_totalHash[threadIdx.x], s_totalHash[threadIdx.x + s], SHA256_BLOCK_SIZE);
                memcpy(s_inputString[threadIdx.x], s_inputString[threadIdx.x + s], 4 * sizeof(int));
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        memcpy(&d_totalHash[blockIdx.x * SHA256_BLOCK_SIZE], s_totalHash[0], SHA256_BLOCK_SIZE);
        memcpy(&d_inputString[blockIdx.x * 4], s_inputString[0], 4 * sizeof(int));
    }
}

int main() {
    BYTE *d_totalHash;
    int *d_inputString;

    cudaMalloc((void **)&d_totalHash, GRID_SIZE * SHA256_BLOCK_SIZE * sizeof(BYTE));
    cudaMalloc((void **)&d_inputString, GRID_SIZE * 3 * sizeof(int));

    setup<<<1, 1>>>();

    int epoch = 0;
    while (epoch < 10) {
        auto start = std::chrono::high_resolution_clock::now();

        kernel<<<GRID_SIZE, BLOCK_SIZE>>>(epoch, d_totalHash, d_inputString);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        double elapsedMillis =
            std::chrono::duration<double, std::milli>(end - start).count();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        BYTE smallestHash[SHA256_BLOCK_SIZE];
        cudaMemcpy(smallestHash, d_totalHash, SHA256_BLOCK_SIZE, cudaMemcpyDeviceToHost);

        int smallestInputString[4];
        cudaMemcpy(smallestInputString, d_inputString, 4 * sizeof(int), cudaMemcpyDeviceToHost);

        char input[39];

        int threadIdx = smallestInputString[0];
        int blockIdx = smallestInputString[1];
        int seed = smallestInputString[2];
        int smallest_epoch = smallestInputString[3];

        input_string(smallest_epoch, threadIdx, blockIdx, seed, input);
       
        printf("%d: ", epoch);
        print_hash(smallestHash);
        printf(" %s\n", input);

        printf("elapsed time: %f\n", elapsedMillis);
        
        float hashes = (float)GRID_SIZE * BLOCK_SIZE * ITEMS_PER_THREAD;
        printHashesPerSecond(hashes, elapsedMillis);

        fflush(stdout);

        epoch++;
    }

    cudaFree(d_totalHash);
    cudaFree(d_inputString);

    return 0;
}