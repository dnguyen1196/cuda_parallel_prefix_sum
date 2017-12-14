#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdlib>
#include <vector>
#include "ee193_utils.hxx"
#include <curand_mtgp32_kernel.h>
#include "helper_cuda.h"
#include "macros.hxx"

using namespace std;

// Sequential scan function
void sequential_scan(int* input, int* output, int num);

__global__ void naive_parallel_scan(int* input, int* output, int num);

__global__ void better_parallel_scan(int* input, int* output, int num);

__global__ void conflict_free_parallel_scan(int* input, int* output, int num);

const int B = 256;

void test_sequential(int *input, int* output, int num);

void test_naive_parallel(int *input, int* output, int num);

void test_better_parallel(int* input, int* output, int num);

void test_conflict_free_parallel_scan(int* input, int* output, int num);

void check_results(int* input, int* output, int num);

int main() {
    // for (int N = 10; N < 28; N += 2) {
    int N = 20;
        int num = 1 << N;
        LOG("Working with " << num << " elements");
        // Generate a random array of length n all with values between 1 and 10
        vector<int> input(num);
        vector<int> output(num+1);
        for (int i = 0; i < num; i++) {
            input[i] = (rand() % 10) + 1;
        }

        // NOTE: the function to check the results have been commented out 
        // as sequential checking takes too much time on big input
        // test_sequential(&input[0], &output[0], num);
        // check_results(&input[0], &output[0], num);
        // std::fill(output.begin(), output.end(), 0);

        // Perform naive parallel scan and time it
        test_naive_parallel(&input[0], &output[0], num);
        // check_results(&input[0], &output[0], num);
        // std::fill(output.begin(), output.end(), 0);

        // Perform better parallel scan and time it
        // test_better_parallel(&input[0], &output[0], num);
        // check_results(&input[0], &output[0], num);
        // std::fill(output.begin(), output.end(), 0);

        // test_conflict_free_parallel_scan(&input[0], &output[0], num);
        // check_results(&input[0], &output[0], num);
    // }
}

void check_results(int* input, int* output, int num) {
    if (output[0] != 0) {
        DIE("Index " << 0 << " expect " << 0 << " got " << output[0]);
    }
    for (int i = 0; i < num; i++) {
        if (output[i+1] != input[i] + output[i]) {
            DIE("Index " << i+1 << " expect " << input[i] + output[i] 
                << " got " << output[i+1]);
        }
    }
}

void test_sequential(int* input, int* output, int num) {
    auto sequential_start = start_time();
    sequential_scan(input, output, num);
    long int time = delta_usec (sequential_start);
    LOG(time/1000000.0 << " seconds to do sequential scan");
}


void test_naive_parallel(int *input, int *output, int num) {
    auto start = start_time();
    int size_in_bytes = num*4;
    int *input_gpu = NULL;
    cudaError_t err = cudaMalloc((void **)&input_gpu, size_in_bytes);
    ERR_CHK (err, "Failed to allocate device input gpu");
    cudaMemcpy(&input_gpu, &input, size_in_bytes, cudaMemcpyHostToDevice); // Copy memory to GPU

    int *output_gpu = NULL;
    err = cudaMalloc((void**)&output_gpu, size_in_bytes+4);
    ERR_CHK (err, "Failed to allocate device output gpu");

    int NumBlocks = num/B;
    int NumThreads = B/2;
    dim3 thBlocks(NumBlocks), threads(NumThreads);

    better_parallel_scan<<<thBlocks, threads>>>(input_gpu, output_gpu, num);

    cudaMemcpy(&output, &output_gpu, size_in_bytes+4, cudaMemcpyDeviceToHost);
    cudaFree(input_gpu);
    cudaFree(output_gpu);

    long int time = delta_usec (start);
    LOG(time/1000000.0 << " seconds to do naive parallel scan");
}

void test_better_parallel(int* input, int* output, int num) {
    auto start = start_time();
    int size_in_bytes = num*4;
    int *input_gpu = NULL;

    cudaError_t err = cudaMalloc((void **)&input_gpu, size_in_bytes);
    ERR_CHK (err, "Failed to allocate device input gpu - better");
    cudaMemcpy(input_gpu, input, size_in_bytes, cudaMemcpyHostToDevice); // Copy memory to GPU

    int *output_gpu = NULL;
    err = cudaMalloc((void**)&output_gpu, size_in_bytes+4);
    ERR_CHK (err, "Failed to allocate device output gpu - better");

    int NumBlocks = num/B;
    int NumThreads = B/2;
    dim3 thBlocks(NumBlocks), threads(NumThreads);

    better_parallel_scan<<<thBlocks, threads>>>(input_gpu, output_gpu, num);

    cudaMemcpy(output, output_gpu, size_in_bytes+4, cudaMemcpyDeviceToHost);

    cudaFree(input_gpu);
    cudaFree(output_gpu);
    long int time = delta_usec (start);
    LOG(time/1000000.0 << " seconds to do a better parallel scan");
}

void test_conflict_free_parallel_scan(int* input, int* output, int num) {
    auto start = start_time();
    int size_in_bytes = num*4;
    int *input_gpu = NULL;
    cudaError_t err = cudaMalloc((void **)&input_gpu, size_in_bytes);
    ERR_CHK (err, "Failed to allocate device input gpu - conflict free");

    cudaMemcpy(input_gpu, input, size_in_bytes, cudaMemcpyHostToDevice); // Copy memory to GPU

    int *output_gpu = NULL;
    err = cudaMalloc((void**)&output_gpu, size_in_bytes+4);
    ERR_CHK (err, "Failed to allocate device output gpu - conflict free");


    int NumBlocks = num/B;
    int NumThreads = B/2;
    dim3 thBlocks(NumBlocks), threads(NumThreads);

    better_parallel_scan<<<thBlocks, threads>>>(input_gpu, output_gpu, num);

    cudaMemcpy(output, output_gpu, size_in_bytes+4, cudaMemcpyDeviceToHost); // Copy memory to GPU
    
    cudaFree(input_gpu);
    cudaFree(output_gpu);

    long int time = delta_usec (start);
    LOG(time/1000000.0 << " seconds to do a conflict free parallel scan");
}


/*
 * Function to perform sequential scan
 */
void sequential_scan(int* input, int* output, int length) {
    output[0] = 0;
    for(int i = 0; i < length; i++) {
        output[i+1] = output[i] + input[i];
    }
}

/*
 * Naive parallel scan
 */
__global__ void naive_parallel_scan(int* input, int* output, int n) {
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
__global__ void better_parallel_scan(int* input, int* output, int n) {
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


/*
 * Conflict free + still efficient parallel scan
 */
__global__ void conflict_free_parallel_scan(int* input, int* output, int n) {
    extern __shared__ int temp[];
    int thid = threadIdx.x;
    int offset = 1;

    int ai = thid;
    int bi = thid + (n/2);

    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi); // Why both are ais?

    temp[ai + bankOffsetA] = input[ai];
    temp[bi + bankOffsetB] = input[bi];

    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0) temp[n-1+CONFLICT_FREE_OFFSET(n-1)] = 0;
    
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset *(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    output[ai] = temp[ai + bankOffsetA];
    output[bi] = temp[bi + bankOffsetB];
}