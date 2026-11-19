#include "cublas/cublas_matmul.h"

#include <cstdio>
#include <cstdlib>

namespace {

void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS: %s\n", cublasGetStatusString(status));
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

CublasMatmul::CublasMatmul(MatmulShape shape) : GpuMatmul(shape) {
    check(cublasCreate(&handle));
    check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
}

CublasMatmul::~CublasMatmul() {
    cublasDestroy(handle);
}

void CublasMatmul::launch(const float *a, const float *b, float *c) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major buffers are transposed views in the cuBLAS column-major layout.
    check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, shape.n, shape.m, shape.k,
                      &alpha, b, shape.n, a, shape.k, &beta, c, shape.n));
}
