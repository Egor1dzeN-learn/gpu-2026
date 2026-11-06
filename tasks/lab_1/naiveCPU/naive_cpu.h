#ifndef GPU_2026_NAIVE_CPU_H
#define GPU_2026_NAIVE_CPU_H

#include <chrono>
#include <cmath>

#include "common/softmax.h"

class NaiveCPU : public AbstractSoftmax{
    SoftmaxResult run(float *input, size_t w, size_t h) override {
        std::vector<float> results(w*h);

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0;i<h;++i){
            float sum = 0;
            for (int j = 0;j<w;++j){
                sum += std::exp(input[i * w + j]);
            }
            for (int j = 0;j<w;++j){
                results[i * w + j] = std::exp(input[i * w + j]) / sum;
            }
        }
        const auto end = std::chrono::steady_clock::now();

        const float ms = std::chrono::duration<float, std::milli>(end - start).count();
        return {std::move(results), ms, ms};
    }
};
#endif
