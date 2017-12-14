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

const int B = 512;

void test_sequential(int *input, int* output, int num);

void RUN(int *input, int* output, int num, int choice);

void check_results(int* input, int* output, int num);

/*
 * Naive parallel scan
 */
__global__ void naive_parallel_scan(int* input, int* output, int n) {
    __shared__ int temp[64]; // This is like a double buffer
    // At any one point, we need sort of two buffer arrays
    // So just keep them both in 1 array and keep track of the indices

    int thid = threadIdx.x;
    int pout = 0; // This will be used to alternate between the two buffer array
    int pin = 1;

    // First load into the shared memory
    // The first element is 0
    temp[pout*n + thid] = (thid > 0) ? input[thid-1] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;
        if (thid > offset) { // If the thread can do work (remember after each level, some part of the array doesn't require
            // further changes
            temp[pout*n + thid] += temp[pin * n + thid - offset];
        } else { // Just copy the data over
            temp[pout*n + thid] = temp[pin * n + thid];
        }
        __syncthreads();
    }

    output[thid] = temp[pout * n + thid]; // Write back on the output
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


void RUN(int *input, int *output, int num, int choice) {
    auto start = start_time();
    int size_in_bytes = num*4;
    int *input_gpu = NULL;
    cudaError_t err = cudaMalloc((void **)&input_gpu, size_in_bytes);
    ERR_CHK (err, "Failed to allocate device input gpu");
    cudaMemcpy((void*)input_gpu, (void*)input, size_in_bytes, cudaMemcpyHostToDevice); // Copy memory to GPU

    int *output_gpu = NULL;
    err = cudaMalloc((void**)&output_gpu, size_in_bytes);
    ERR_CHK (err, "Failed to allocate device output gpu");

    int NumBlocks = num/B;
    int NumThreads = B;
    int NumBytesShared;
    dim3 thBlocks(NumBlocks), threads(NumThreads);

    string name = "";
    if (choice == 1){
        NumBytesShared = size_in_bytes * 2;
        naive_parallel_scan<<<thBlocks, threads>>>(input_gpu, output_gpu, num);
        name = "naive parallel";
    } else if (choice == 2) {
        better_parallel_scan<<<thBlocks, threads>>>(input_gpu, output_gpu, num);
        name = "better parallel";
    } else {
        conflict_free_parallel_scan<<<thBlocks, threads>>>(input_gpu, output_gpu, num);
        name = "conflict free parallel";
    }

    cudaMemcpy((void*)output, (void*)output_gpu, size_in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(input_gpu);
    cudaFree(output_gpu);

    long int time = delta_usec (start);

    LOG(time/1000000.0 << " seconds to do " << name << " scan");
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

int main() {
        srand(317);
    // for (int N = 10; N < 28; N += 2) {
        int N = 5;
        int num = 1 << N;
        LOG("Working with " << num << " elements");
        // Generate a random array of length n all with values between 1 and 10
        vector<int> input(num);
        vector<int> output(num+1);
        for (int i = 0; i < num; i++) {
            input[i] = (rand() % 100) + 1;
        }

        for (int i = 0; i < 5; i++) {
            cout << input[i] << endl;
        }
        // NOTE: the function to check the results have been commented out 
        // as sequential checking takes too much time on big input
        // test_sequential(&input[0], &output[0], num);
        // check_results(&input[0], &output[0], num);
        std::fill(output.begin(), output.end(), 0);

        // Perform naive parallel scan and time it
        RUN(&input[0], &output[0], num, 2);
        for (int i = 0; i < 5; i++) {
            cout << output[i] << endl;
        }
}