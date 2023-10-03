// histogram.cu — GPU Histogram with Shared Memory Atomics
// Uses per-block shared memory histogram + global merge to avoid atomic contention.

#include <cuda_runtime.h>
#include <stdio.h>

#define NUM_BINS  256
#define BLOCK_SIZE 256

__global__ void histogramShared(const unsigned char* __restrict__ data,
                                 unsigned int* __restrict__ histo, int N)
{
    __shared__ unsigned int local_histo[NUM_BINS];

    // Initialize shared histogram
    if (threadIdx.x < NUM_BINS)
        local_histo[threadIdx.x] = 0;
    __syncthreads();

    // Each thread processes multiple elements (grid-stride loop)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < N) {
        atomicAdd(&local_histo[data[idx]], 1);
        idx += stride;
    }
    __syncthreads();

    // Merge local into global histogram
    if (threadIdx.x < NUM_BINS)
        atomicAdd(&histo[threadIdx.x], local_histo[threadIdx.x]);
}

int main() {
    const int N = 1 << 22;
    unsigned char* h = new unsigned char[N];
    for (int i = 0; i < N; i++) h[i] = i % NUM_BINS;

    unsigned char* d_data;
    unsigned int*  d_histo;
    cudaMalloc(&d_data,  N * sizeof(unsigned char));
    cudaMalloc(&d_histo, NUM_BINS * sizeof(unsigned int));
    cudaMemset(d_histo, 0, NUM_BINS * sizeof(unsigned int));
    cudaMemcpy(d_data, h, N * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    histogramShared<<<blocks, BLOCK_SIZE>>>(d_data, d_histo, N);

    unsigned int h_histo[NUM_BINS];
    cudaMemcpy(h_histo, d_histo, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Bin[0]=%u (expected %u)\n", h_histo[0], N / NUM_BINS);

    cudaFree(d_data); cudaFree(d_histo); delete[] h;
    return 0;
}
