#ifndef GPU_2026_CUDA_MATMUL_H
#define GPU_2026_CUDA_MATMUL_H

#include "common/matmul.h"

#include <cstddef>
#include <cuda_runtime_api.h>

class GpuMatmul : public AbstractMatmul {
public:
    explicit GpuMatmul(MatmulShape shape);
    ~GpuMatmul() override;
    GpuMatmul(const GpuMatmul &) = delete;
    GpuMatmul &operator=(const GpuMatmul &) = delete;

    MatmulTime run(const float *a, const float *b, float *c) final;

protected:
    virtual void launch(const float *a, const float *b, float *c) = 0;

private:
    std::size_t bytes_a;
    std::size_t bytes_b;
    std::size_t bytes_c;
    float *device_a;
    float *device_b;
    float *device_c;
    cudaEvent_t start;
    cudaEvent_t stop;
};

class CudaNaiveMatmul final : public GpuMatmul {
public:
    using GpuMatmul::GpuMatmul;

private:
    void launch(const float *a, const float *b, float *c) override;
};

class CudaSharedMatmul final : public GpuMatmul {
public:
    using GpuMatmul::GpuMatmul;

private:
    void launch(const float *a, const float *b, float *c) override;
};

class CudaRegisterMatmul final : public GpuMatmul {
public:
    using GpuMatmul::GpuMatmul;

private:
    void launch(const float *a, const float *b, float *c) override;
};

#endif
