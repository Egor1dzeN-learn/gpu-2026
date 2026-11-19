#include "cudnn/cudnn_softmax.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <cuda_runtime.h>

namespace {

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in cuDNN benchmark: %s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

void check_cudnn(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuDNN error: %s\n", cudnnGetErrorString(status));
        std::exit(1);
    }
}

} // namespace

CudnnSoftmax::CudnnSoftmax() {
    check_cudnn(cudnnCreate(&handle));
    check_cudnn(cudnnCreateTensorDescriptor(&tensor));
}

CudnnSoftmax::~CudnnSoftmax() {
    cudnnDestroyTensorDescriptor(tensor);
    cudnnDestroy(handle);
}

SoftmaxResult CudnnSoftmax::run(float *input, size_t w, size_t h) {
    // Treat each row as a sample and its columns as channels.
    check_cudnn(cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         static_cast<int>(h), static_cast<int>(w), 1, 1));
    const auto begin = std::chrono::steady_clock::now();
    const size_t bytes = w * h * sizeof(float);
    float *gpu_in = nullptr, *gpu_out = nullptr;
    check_cuda(cudaMalloc(&gpu_in, bytes));
    check_cuda(cudaMalloc(&gpu_out, bytes));
    check_cuda(cudaMemcpy(gpu_in, input, bytes, cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));
    check_cuda(cudaEventRecord(start));
    check_cudnn(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                   &alpha, tensor, gpu_in, &beta, tensor, gpu_out));
    check_cuda(cudaEventRecord(stop));
    check_cuda(cudaEventSynchronize(stop));
    float kernel_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&kernel_ms, start, stop));
    check_cuda(cudaEventDestroy(start));
    check_cuda(cudaEventDestroy(stop));

    std::vector<float> result(w * h);
    check_cuda(cudaMemcpy(result.data(), gpu_out, bytes, cudaMemcpyDeviceToHost));
    check_cuda(cudaFree(gpu_in));
    check_cuda(cudaFree(gpu_out));
    const auto end = std::chrono::steady_clock::now();
    const float total_ms = std::chrono::duration<float, std::milli>(end - begin).count();
    return {std::move(result), total_ms, kernel_ms};
}
