#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <chrono>

#include "sha256.cuh"
#include "helper.cuh"

#define BLOCK_SIZE 1024
#define GRID_SIZE 5120
#define ITEMS_PER_THREAD 32

const char *h_baseline = "telaxion/zig+2070super+";
__device__ const char *d_baseline = "telaxion/zig+2070super+";
__device__ const int baseline_len = 23;

__both__ int input_string(int epoch, int tid, int bid, int item,
                          char *output) {
    int len = 0;

    #if defined(__CUDA_ARCH__)
        memcpy(output, d_baseline, baseline_len);
    #else
        memcpy(output, h_baseline, baseline_len);
    #endif    

    len += baseline_len;

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


__device__ BYTE s_totalHash[BLOCK_SIZE][SHA256_BLOCK_SIZE];
__device__ int s_inputString[BLOCK_SIZE][3];

__device__ inline void findHash(int epoch, int threadIdx, int blockIdx, int seed, BYTE *smallestHash, int *smallestInput) {
    char input[baseline_len + 16];
    int len = input_string(epoch, threadIdx, blockIdx, seed, input);

    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (BYTE *)input, len);

    BYTE digest[SHA256_BLOCK_SIZE];
    sha256_final(&ctx, digest);

    int compare = memcmpHash<32>(digest, smallestHash);
    if (compare < 0) {
        memcpy(smallestHash, digest, SHA256_BLOCK_SIZE);
        smallestInput[0] = threadIdx;
        smallestInput[1] = blockIdx;
        smallestInput[2] = seed;
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
        memcpy(s_smallestInput, smallestInput, 3 * sizeof(int));
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            int compare = memcmpHash<32>(s_totalHash[threadIdx.x], s_totalHash[threadIdx.x + s]);
            if (compare > 0) {
                memcpy(s_totalHash[threadIdx.x], s_totalHash[threadIdx.x + s], SHA256_BLOCK_SIZE);
                memcpy(s_inputString[threadIdx.x], s_inputString[threadIdx.x + s], 3 * sizeof(int));
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        memcpy(&d_totalHash[blockIdx.x * SHA256_BLOCK_SIZE], s_totalHash[0], SHA256_BLOCK_SIZE);
        memcpy(&d_inputString[blockIdx.x * 3], s_inputString[0], 3 * sizeof(int));
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

        int smallestInputString[3];
        cudaMemcpy(smallestInputString, d_inputString, 3 * sizeof(int), cudaMemcpyDeviceToHost);

        char input[baseline_len + 16];
        input_string(epoch, smallestInputString[0], smallestInputString[1], smallestInputString[2], input);
       
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
