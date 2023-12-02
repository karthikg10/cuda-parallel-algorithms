// bitonic_sort.cu — Parallel Bitonic Sort on GPU
// O(n log^2 n) comparisons, fully parallelized across threads.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void bitonicSortStep(float* arr, int j, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int ixj = idx ^ j;  // XOR partner

    if (ixj > idx) {
        bool ascending = ((idx & k) == 0);
        if (ascending == (arr[idx] > arr[ixj])) {
            // Swap
            float tmp  = arr[idx];
            arr[idx]   = arr[ixj];
            arr[ixj]   = tmp;
        }
    }
}

void bitonicSort(float* d_arr, int N) {
    int threads = BLOCK_SIZE;
    int blocks  = (N + threads - 1) / threads;

    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortStep<<<blocks, threads>>>(d_arr, j, k);
            cudaDeviceSynchronize();
        }
    }
}

int main() {
    const int N = 1024;  // Must be power of 2
    float* h = new float[N];
    srand(42);
    for (int i = 0; i < N; i++) h[i] = (float)rand() / RAND_MAX;

    float* d;
    cudaMalloc(&d, N * sizeof(float));
    cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice);

    bitonicSort(d, N);

    cudaMemcpy(h, d, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify sorted
    bool sorted = true;
    for (int i = 1; i < N; i++)
        if (h[i] < h[i-1]) { sorted = false; break; }
    printf("Bitonic sort N=%d: %s\n", N, sorted ? "PASSED" : "FAILED");

    cudaFree(d); delete[] h;
    return 0;
}
