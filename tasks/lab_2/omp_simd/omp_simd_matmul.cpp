#include "omp_simd/omp_simd_matmul.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <immintrin.h>

MatmulTime OmpSimdMatmul::run(const float *a, const float *b, float *c) {
    const auto start = std::chrono::steady_clock::now();
    constexpr int row_block = 16;
    constexpr int depth_block = 64;
    constexpr int col_block = 128;
    const int blocks = (shape.m + row_block - 1) / row_block;

#pragma omp parallel for schedule(static) num_threads(threads)
    for (int block = 0; block < blocks; ++block) {
        const int first_row = block * row_block;
        const int last_row = std::min(first_row + row_block, shape.m);
        std::fill(c + static_cast<size_t>(first_row) * shape.n,
                  c + static_cast<size_t>(last_row) * shape.n, 0.0f);

        for (int col = 0; col < shape.n; col += col_block) {
            const int last_col = std::min(col + col_block, shape.n);
            for (int depth = 0; depth < shape.k; depth += depth_block) {
                const int last_depth = std::min(depth + depth_block, shape.k);
                for (int i = first_row; i < last_row; ++i) {
                    const float *a_row = a + static_cast<size_t>(i) * shape.k;
                    float *c_row = c + static_cast<size_t>(i) * shape.n;
                    for (int k = depth; k < last_depth; ++k) {
                        const float *b_row = b + static_cast<size_t>(k) * shape.n;
                        const __m256 value = _mm256_set1_ps(a_row[k]);
                        int j = col;
                        for (; j + 8 <= last_col; j += 8) {
                            const __m256 sum = _mm256_loadu_ps(c_row + j);
                            const __m256 rhs = _mm256_loadu_ps(b_row + j);
                            _mm256_storeu_ps(c_row + j, _mm256_fmadd_ps(value, rhs, sum));
                        }
                        for (; j < last_col; ++j) {
                            c_row[j] += a_row[k] * b_row[j];
                        }
                    }
                }
            }
        }
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return {std::chrono::duration<double, std::milli>(elapsed).count(), 0.0f};
}
