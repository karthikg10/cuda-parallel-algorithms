// prefix_scan.cu — Exclusive Prefix Scan (Blelloch Algorithm)
// Work-efficient parallel scan: O(n) work, O(log n) depth.

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 512

// Single-block exclusive scan (Blelloch up-sweep + down-sweep)
__global__ void exclusiveScan(float* data, int N) {
    __shared__ float temp[BLOCK_SIZE * 2];

    int tid = threadIdx.x;
    int offset = 1;

    // Load into shared memory
    temp[2 * tid]     = (2 * tid     < N) ? data[2 * tid]     : 0;
    temp[2 * tid + 1] = (2 * tid + 1 < N) ? data[2 * tid + 1] : 0;

    // Up-sweep (reduce) phase
    for (int d = BLOCK_SIZE; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Set identity at root
    if (tid == 0) temp[BLOCK_SIZE * 2 - 1] = 0;

    // Down-sweep phase
    for (int d = 1; d < BLOCK_SIZE * 2; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t   = temp[ai];
            temp[ai]  = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (2 * tid     < N) data[2 * tid]     = temp[2 * tid];
    if (2 * tid + 1 < N) data[2 * tid + 1] = temp[2 * tid + 1];
}

int main() {
    const int N = 16;
    float h[N];
    for (int i = 0; i < N; i++) h[i] = (float)(i + 1);

    float* d;
    cudaMalloc(&d, N * sizeof(float));
    cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice);

    exclusiveScan<<<1, BLOCK_SIZE>>>(d, N);
    cudaMemcpy(h, d, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Exclusive scan output:\n");
    for (int i = 0; i < N; i++) printf("%.0f ", h[i]);
    printf("\n(Expected: 0 1 3 6 10 15 21 28 36 45 55 66 78 91 105 120)\n");

    cudaFree(d);
    return 0;
}
