#include "naiveCPU/naive_cpu.h"
#include "common/cpu_row.h"

#include <chrono>

MatmulTime NaiveMatmul::run(const float *a, const float *b, float *c) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < shape.m; ++i) {
        matmul_row(a + static_cast<size_t>(i) * shape.k, b, c + static_cast<size_t>(i) * shape.n, shape);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return {std::chrono::duration<double, std::milli>(elapsed).count(), 0.0f};
}
