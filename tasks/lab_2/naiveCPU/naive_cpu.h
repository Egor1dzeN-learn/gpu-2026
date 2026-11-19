#ifndef GPU_2026_NAIVE_MATMUL_H
#define GPU_2026_NAIVE_MATMUL_H

#include "common/matmul.h"

class NaiveMatmul final : public AbstractMatmul {
public:
    using AbstractMatmul::AbstractMatmul;
    MatmulTime run(const float *a, const float *b, float *c) override;
};

#endif
