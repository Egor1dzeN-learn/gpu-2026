#ifndef GPU_2026_CUBLAS_MATMUL_H
#define GPU_2026_CUBLAS_MATMUL_H

#include "cuda/cuda_matmul.h"

#include <cublas_v2.h>

class CublasMatmul final : public GpuMatmul {
public:
    explicit CublasMatmul(MatmulShape shape);
    ~CublasMatmul() override;

private:
    void launch(const float *a, const float *b, float *c) override;

    cublasHandle_t handle;
};

#endif
