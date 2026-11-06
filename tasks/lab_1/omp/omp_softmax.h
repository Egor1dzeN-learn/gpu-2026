#ifndef GPU_2026_OMP_SOFTMAX_H
#define GPU_2026_OMP_SOFTMAX_H

#include <chrono>
#include <cmath>
#include <omp.h>

#include "common/softmax.h"

class OmpSoftmax : public AbstractSoftmax {
private:
    [[maybe_unused]] int count_thread = omp_get_max_threads();
public:
    explicit OmpSoftmax(int count_thread = omp_get_max_threads()) {
        this->count_thread = count_thread;
    }

    SoftmaxResult run(float *input, size_t w, size_t h) override {
        std::vector<float> results(w * h);

        const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static) num_threads(this->count_thread)
        for (int i = 0; i < h; ++i) {
            float sum = 0;
            for (int j = 0; j < w; ++j) {
                sum += std::exp(input[i * w + j]);
            }
            for (int j = 0; j < w; ++j) {
                results[i * w + j] = std::exp(input[i * w + j]) / sum;
            }
        }
        const auto end = std::chrono::steady_clock::now();

        const float ms = std::chrono::duration<float, std::milli>(end - start).count();
        return {std::move(results), ms, ms};
    }
};

#endif
