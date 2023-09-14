// parallel_reduction.cu — GPU Parallel Reduction (Sum)
// Shared memory tree reduction with warp-level unrolling.

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void reduceSum(const float* __restrict__ input,
                          float* __restrict__ output, int N)
{
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float val = 0.0f;
    if (idx < N)                val  = input[idx];
    if (idx + blockDim.x < N)   val += input[idx + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Warp-level unrolled (no __syncthreads within a warp)
    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid +  8];
        smem[tid] += smem[tid +  4];
        smem[tid] += smem[tid +  2];
        smem[tid] += smem[tid +  1];
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

float parallelSum(const float* d_input, int N) {
    int threads = BLOCK_SIZE;
    int blocks  = (N + threads * 2 - 1) / (threads * 2);
    float *d_partial, *d_out;
    cudaMalloc(&d_partial, blocks * sizeof(float));
    cudaMalloc(&d_out,     sizeof(float));
    reduceSum<<<blocks, threads>>>(d_input, d_partial, N);
    if (blocks > 1)
        reduceSum<<<1, threads>>>(d_partial, d_out, blocks);
    else
        cudaMemcpy(d_out, d_partial, sizeof(float), cudaMemcpyDeviceToDevice);
    float result;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_partial); cudaFree(d_out);
    return result;
}

int main() {
    const int N = 1 << 20;
    float* h = new float[N];
    for (int i = 0; i < N; i++) h[i] = 1.0f;
    float* d; cudaMalloc(&d, N * sizeof(float));
    cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice);
    printf("Sum (expected %d): %.0f\n", N, parallelSum(d, N));
    cudaFree(d); delete[] h;
    return 0;
}
