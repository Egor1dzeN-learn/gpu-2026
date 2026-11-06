#include "cuda/cuda_softmax.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",               \
                         cudaGetErrorString(err_), __FILE__, __LINE__);    \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

__global__ void softmax_kernel(const float *input, float *out, int w, int h) {
    const float *row_in = input + (size_t) blockIdx.x * w;
    float *row_out = out + (size_t) blockIdx.x * w;
    __shared__ float sum_[256];
    float part_sum = 0;

    for (unsigned int i = threadIdx.x; i < w; i += blockDim.x) {
        row_out[i] = expf(row_in[i]);
        part_sum += row_out[i];
    }
    sum_[threadIdx.x] = part_sum;

    __syncthreads();

    float sum = 0;
    for (float i: sum_) {
        sum += i;
    }

    __syncthreads();

    for (unsigned int i = threadIdx.x; i < w; i += blockDim.x) {
        row_out[i] = row_out[i] / sum;
    }
}

std::vector<float> softmax_kernel(float *input, size_t w, size_t h, float *kernel_ms) {
    const size_t input_bytes = w * h * sizeof(float);
    float *gpu_in = nullptr, *gpu_out = nullptr;
    CUDA_CHECK(cudaMalloc(&gpu_in, input_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_out, input_bytes));
    CUDA_CHECK(cudaMemcpy(gpu_in, input, input_bytes, cudaMemcpyHostToDevice));

    const int block = (int) 256;
    const int grid = (int) h;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel<<<grid, block >>>(gpu_in, gpu_out, (int) w, (int) h);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(kernel_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::vector<float> result(w * h);
    CUDA_CHECK(cudaMemcpy(result.data(), gpu_out, input_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(gpu_in));
    CUDA_CHECK(cudaFree(gpu_out));
    return result;
}