#ifndef GPU_2026_OMP_SIMD_SOFTMAX_H
#define GPU_2026_OMP_SIMD_SOFTMAX_H

#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <omp.h>

#include "common/softmax.h"

class OmpSIMDSoftmax : public AbstractSoftmax {
    SoftmaxResult run(float *input, size_t w, size_t h) override {
        std::vector<float> results(w * h);
        const auto start = std::chrono::steady_clock::now();

#pragma omp parallel for schedule(static)
        for (int i = 0; i < h; ++i) {
            const float *row_in = input + i * w;
            float *row_out = results.data() + i * w;

            __m256 acc = _mm256_setzero_ps();
            size_t j = 0;
            for (; j + 8 <= w; j += 8) {
                const __m256 e = _mm256_exp_ps(_mm256_loadu_ps(row_in + j));
                _mm256_storeu_ps(row_out + j, e);
                acc = _mm256_add_ps(acc, e);
            }

            float lanes[8];
            _mm256_storeu_ps(lanes, acc);
            float sum = 0.0f;
            for (float lane: lanes)
                sum += lane;
            for (; j < w; ++j) {
                row_out[j] = std::exp(row_in[j]);
                sum += row_out[j];
            }

            const float inv_sum = 1.0f / sum;
            const __m256 inv = _mm256_set1_ps(inv_sum);
            for (j = 0; j + 8 <= w; j += 8)
                _mm256_storeu_ps(row_out + j, _mm256_mul_ps(_mm256_loadu_ps(row_out + j), inv));
            for (; j < w; ++j)
                row_out[j] *= inv_sum;

            _mm256_zeroupper();
        }

        const auto end = std::chrono::steady_clock::now();
        const float ms = std::chrono::duration<float, std::milli>(end - start).count();
        return {std::move(results), ms, ms};
    }
};

#endif
