#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdlib>
#include "ee193_utils.hxx
#include <curand_mtgp32_kernel.h>
#include "helper_cuda.h"

// Sequential scan function
void sequential_scan(int* input, int* output, int length);

__global__ void naive_parallel_scan(int* input, int* output, int length);

__global__ void better_parallel_scan(int* input, int* output, int length);

const int B = 512;

int main() {
    int N = 10;
    int num = N >> 1;

    // Generate a random array of length n all with values between 1 and 10
    int input[num];
    int output[num+1];
    for (int i = 0; i < num; i++) {
        input[i] = (rand() % 10) + 1;
    }

    // Perform sequential scan and time it
    auto sequential_start = start_time();
    sequential_scan(output, input, num);
    long int time = delta_usec (sequential_start);
    cout << time/1000000.0 << " seconds to do sequential scan" << endl;

    // Perform naive parallel scan and time it
    auto naive_start = start_time();
    int size_in_bytes = num*4;
    int *input_naive = NULL;
    cudaError_t err = cudaMalloc((void **)&input_naive, size_in_bytes);
    cudaMemcpy(input_naive, input, size_in_bytes, cudaMemcpyHostToDevice); // Copy memory to GPU

    int *output_naive = NULL;
    err = cudaMalloc((void**)&output_naive, size_in_bytes);

    int NumBlocks = num/B;
    int NumThreads = B/2;
    dim3 thBlocks(NumBlocks), threads(NumThreads);
    naive_parallel_scan<<<thBlocks, threads>>>(input_naive, output_naive, num);
    cudaMemcpy(output, output_naive, size_in_bytes+4, cudaMemcpyDeviceToHost); // Copy memory to GPU
    long int time = delta_usec (naive_start);
    cout << time/1000000.0 << " seconds to do naive parallel scan" << endl;
    // Perform better parallel scan and time it

}

/*
 * Function to perform sequential scan
 */
void sequential_scan(int* output, int* input, int length) {
    output[0] = 0;
    for(int i = 0; i < length; i++) {
        output[i+1] = output[i] + input[i];
    }
}

/*
 * Naive parallel scan
 */
__global__ void naive_parallel_scan(int* output, int* input, int n) {
    extern __shared__ int shared_mem[]; // This is like a double buffer
    // At any one point, we need sort of two buffer arrays
    // So just keep them both in 1 array and keep track of the indices

    int thid = threadIdx.x;
    int pout = 0; // This will be used to alternate between the two buffer array
    int pin = 1;

    // First load into the shared memory
    // The first element is 0
    shared_mem[pout*n + thid] = (thid > 0) ? input[thid-1] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;
        if (thid > offset) { // If the thread can do work (remember after each level, some part of the array doesn't require
            // further changes
            shared_mem[pout*n + thid] += shared_mem[pin * n + thid - offset];
        } else { // Just copy the data over
            shared_mem[pout*n + thid] = shared_mem[pin * n + thid];
        }
        __syncthreads();
    }
    output[thid] = shared_mem[pout * n + thid]; // Write back on the output
}


/*
 * Efficient parallel scan
 */
__global__ void better_parallel_scan(int* output, int* input, int n) {
    extern __shared__ int temp[];
    int thid = threadIdx.x;
    int offset = 1;

    temp[2*thid] = input[2*thid];
    temp[2*thid + 1] = input[2*thid + 1];

    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0) temp[n-1] = 0;
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset *(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    output[2*thid] = temp[2*thid];
    output[2*thid+1] = temp[2*thid+1];
}