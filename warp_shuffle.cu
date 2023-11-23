#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <string>


#define BLOCK_SIZE 32 // Number of threads per block 
#define PRINT_SIZE 50

// Function to perform warp shuffle sum within a warp
__device__ int warpSum(int value);
__global__ void sumReductionKernel(int *input, int* output, int n);
__device__ int blockReduceSum(int val);
void setMatrix(int *matrix, int n, int seed);
void printMatrix(int *matrix, int n);
int main(int argc, char** argv) {
    // Check the number of arguments
    if (argc != 4) {
        fprintf(stderr, "run as ./prog <gpu-id>  n <seed> \n");
        fprintf(stderr, "seed: If seed is '0', initilize the matrix with '1s'");
        exit(EXIT_FAILURE);
    }
    // Initialization
    printf("Initializing.....\n");
    int gpu_id = atoi(argv[1]); // ID of the GPU
    int n = atoi(argv[2]);      // Size problem 
    int seed = atoi(argv[3]); //Seed
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int *h_array = new int[n], *h_result = new int[numBlocks]; 
    int *d_array, *d_result;
    printf("gpu_id: %d, n : %d, seed : %d\n", gpu_id, n, seed);
    setMatrix(h_array, n, seed);
    float msecs = 0.0f;
    if( n < PRINT_SIZE ) {
        printMatrix(h_array, n);
    }
    cudaMalloc(&d_array, n * sizeof(int)); // Assing device memory 
    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);  // Copy array from host to device
    cudaMalloc(&d_result, numBlocks * sizeof(int));
    cudaMemset(d_result, 0, numBlocks * sizeof(int));
    // Create event to measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sumReductionKernel<<<numBlocks, BLOCK_SIZE>>>(d_array, d_result, n);
    cudaMemcpy(h_result, d_result, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Calculate time
    cudaEventElapsedTime(&msecs, start, stop);
    // Sum array results
    int final_result = 0;
    for (int i = 0; i < numBlocks; ++i) {
        final_result += h_result[i];
    }
    printf("Sum of array elements: %d\n", final_result);
    printf("done: time GPU: %f secs\n", msecs/1000.0f);
    cudaFree(d_array);
    cudaFree(d_result);
    delete[] h_array;
    delete[] h_result;
    fflush(stdout);
    printf("El programa termino con exito!\n");
    exit(EXIT_SUCCESS);
}
// Reduce the warp using warp shuffle
__device__ int warpSum(int value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }
    return value;
}

__device__ int blockReduceSum(int val) {

    static __shared__ int shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize; // Warp id

    val = warpSum(val);     // Each warp performs partial reduction

    if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid == 0) val = warpSum(val); // Final reduce within the first warp

    return val;
}

__global__ void sumReductionKernel(int *input, int* output, int n) {
    int sum = 0;
    // Reduce multiple elements per thread
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tidx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0)
        output[blockIdx.x] = sum;
}

void setMatrix(int *matrix, int n, int seed){
    if(seed == 0)
        for(int i = 0; i < n; ++i) matrix[i] = 1;
    else{
        srand(seed);
        for(int i = 0; i < n; ++i) matrix[i] = rand()%2;
    }

}

void printMatrix(int *matrix, int n){
    printf("[");
    for (int i = 0; i < n; i++){
        printf("%d,", matrix[i]);
    }
    printf("]\n");

}