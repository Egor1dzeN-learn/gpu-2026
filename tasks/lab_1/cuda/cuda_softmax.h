#ifndef GPU_2026_CUDA_SOFTMAX_H
#define GPU_2026_CUDA_SOFTMAX_H

#include <chrono>
#include <cmath>
#include <omp.h>

#include "common/softmax.h"

std::vector<float> softmax_kernel(float *input, size_t w, size_t h, float *kernel_ms);

class CudaSoftmax : public AbstractSoftmax {
    SoftmaxResult run(float *input, size_t w, size_t h) override {
        const auto start = std::chrono::steady_clock::now();

        float kernel_ms = 0.0f;
        auto res = softmax_kernel(input, w, h, &kernel_ms);

        const auto end = std::chrono::steady_clock::now();
        return {std::move(res),
                std::chrono::duration<float, std::milli>(end - start).count(),
                kernel_ms};
    }
};

#endif
