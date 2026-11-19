#ifndef GPU_2026_MATMUL_CPU_ROW_H
#define GPU_2026_MATMUL_CPU_ROW_H

#include "common/matmul.h"

#include <algorithm>
#include <cstddef>

inline void matmul_row(const float *a, const float *b, float *__restrict c, MatmulShape shape) {
    std::fill_n(c, shape.n, 0.0f);
    for (int k = 0; k < shape.k; ++k) {
        const float value = a[k];
        const float *b_row = b + static_cast<size_t>(k) * shape.n;
        for (int j = 0; j < shape.n; ++j) {
            c[j] += value * b_row[j];
        }
    }
}

#endif
