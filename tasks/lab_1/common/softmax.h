#ifndef GPU_2026_SOFTMAX_H
#define GPU_2026_SOFTMAX_H

#include <vector>

struct SoftmaxResult {
    std::vector<float> results;
    float execution_time;
    float kernel_time;
};

class AbstractSoftmax {
public:
    virtual SoftmaxResult run(float *input, size_t w, size_t h) {

        return {};
    }
};

#endif
