#ifndef GPU_2026_CUDNN_SOFTMAX_H
#define GPU_2026_CUDNN_SOFTMAX_H

#include <cudnn.h>

#include "common/softmax.h"

class CudnnSoftmax : public AbstractSoftmax {
public:
    CudnnSoftmax();
    ~CudnnSoftmax();
    CudnnSoftmax(const CudnnSoftmax &) = delete;
    CudnnSoftmax &operator=(const CudnnSoftmax &) = delete;

private:
    SoftmaxResult run(float *input, size_t w, size_t h) override;

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t tensor;
};

#endif
