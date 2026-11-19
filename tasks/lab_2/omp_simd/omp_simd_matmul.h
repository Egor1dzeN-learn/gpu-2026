#ifndef GPU_2026_OMP_SIMD_MATMUL_H
#define GPU_2026_OMP_SIMD_MATMUL_H

#include "common/matmul.h"

class OmpSimdMatmul final : public AbstractMatmul {
public:
    OmpSimdMatmul(MatmulShape shape, int threads) : AbstractMatmul(shape), threads(threads) {}
    MatmulTime run(const float *a, const float *b, float *c) override;

private:
    int threads;
};

#endif
