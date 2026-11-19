#ifndef GPU_2026_MATMUL_H
#define GPU_2026_MATMUL_H

struct MatmulShape {
    int m;
    int k;
    int n;
};

struct MatmulTime {
    double total_ms;
    float kernel_ms;
};

class AbstractMatmul {
public:
    explicit AbstractMatmul(MatmulShape shape) : shape(shape) {}
    virtual ~AbstractMatmul() = default;

    virtual MatmulTime run(const float *a, const float *b, float *c) = 0;

protected:
    const MatmulShape shape;
};

#endif
