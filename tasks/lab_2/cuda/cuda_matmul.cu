#include "cuda/cuda_matmul.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace {

constexpr int TILE_SIZE = 16;

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__device__ float read_or_zero(const float *matrix, int row, int col, int rows, int cols) {
    return row < rows && col < cols ? matrix[row * cols + col] : 0.0f;
}

__device__ void write_result(float *c, int row, int col, MatmulShape s, float sum) {
    if (row < s.m && col < s.n) {
        c[row * s.n + col] = sum;
    }
}

__global__ void naive_kernel(const float *a, const float *b, float *c, MatmulShape s) {
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (row >= s.m || col >= s.n) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < s.k; ++k) {
        sum += a[row * s.k + k] * b[k * s.n + col];
    }
    c[row * s.n + col] = sum;
}

__global__ void shared_kernel(const float *a, const float *b, float *c, MatmulShape s) {
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + y;
    const int col = blockIdx.x * TILE_SIZE + x;

    float sum = 0.0f;
    for (int offset = 0; offset < s.k; offset += TILE_SIZE) {
        a_tile[y][x] = read_or_zero(a, row, offset + x, s.m, s.k);
        b_tile[y][x] = read_or_zero(b, offset + y, col, s.k, s.n);
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += a_tile[y][k] * b_tile[k][x];
        }
        __syncthreads();
    }
    write_result(c, row, col, s, sum);
}

__global__ void register_kernel(const float *a, const float *b, float *c, MatmulShape s) {
    __shared__ float a_tile[2 * TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][2 * TILE_SIZE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int row0 = blockIdx.y * 2 * TILE_SIZE + y;
    const int col0 = blockIdx.x * 2 * TILE_SIZE + x;
    const int row1 = row0 + TILE_SIZE;
    const int col1 = col0 + TILE_SIZE;
    float sum00 = 0.0f, sum01 = 0.0f;
    float sum10 = 0.0f, sum11 = 0.0f;

    for (int offset = 0; offset < s.k; offset += TILE_SIZE) {
        a_tile[y][x] = read_or_zero(a, row0, offset + x, s.m, s.k);
        a_tile[y + TILE_SIZE][x] = read_or_zero(a, row1, offset + x, s.m, s.k);
        b_tile[y][x] = read_or_zero(b, offset + y, col0, s.k, s.n);
        b_tile[y][x + TILE_SIZE] = read_or_zero(b, offset + y, col1, s.k, s.n);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            const float a0 = a_tile[y][k];
            const float a1 = a_tile[y + TILE_SIZE][k];
            const float b0 = b_tile[k][x];
            const float b1 = b_tile[k][x + TILE_SIZE];
            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }
        __syncthreads();
    }

    write_result(c, row0, col0, s, sum00);
    write_result(c, row0, col1, s, sum01);
    write_result(c, row1, col0, s, sum10);
    write_result(c, row1, col1, s, sum11);
}

dim3 grid_for(MatmulShape shape, int width) {
    return dim3((shape.n + width - 1) / width, (shape.m + width - 1) / width);
}

} // namespace

GpuMatmul::GpuMatmul(MatmulShape shape)
    : AbstractMatmul(shape),
      bytes_a(sizeof(float) * shape.m * shape.k),
      bytes_b(sizeof(float) * shape.k * shape.n),
      bytes_c(sizeof(float) * shape.m * shape.n) {
    check(cudaMalloc(&device_a, bytes_a));
    check(cudaMalloc(&device_b, bytes_b));
    check(cudaMalloc(&device_c, bytes_c));
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
}

GpuMatmul::~GpuMatmul() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
}

MatmulTime GpuMatmul::run(const float *a, const float *b, float *c) {
    const auto begin = std::chrono::steady_clock::now();
    check(cudaMemcpy(device_a, a, bytes_a, cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_b, b, bytes_b, cudaMemcpyHostToDevice));
    check(cudaEventRecord(start));

    launch(device_a, device_b, device_c);
    check(cudaGetLastError());
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    check(cudaMemcpy(c, device_c, bytes_c, cudaMemcpyDeviceToHost));
    const auto end = std::chrono::steady_clock::now();
    float kernel_ms;
    check(cudaEventElapsedTime(&kernel_ms, start, stop));
    return {std::chrono::duration<double, std::milli>(end - begin).count(), kernel_ms};
}

void CudaNaiveMatmul::launch(const float *a, const float *b, float *c) {
    naive_kernel<<<grid_for(shape, TILE_SIZE), dim3(TILE_SIZE, TILE_SIZE)>>>(a, b, c, shape);
}

void CudaSharedMatmul::launch(const float *a, const float *b, float *c) {
    shared_kernel<<<grid_for(shape, TILE_SIZE), dim3(TILE_SIZE, TILE_SIZE)>>>(a, b, c, shape);
}

void CudaRegisterMatmul::launch(const float *a, const float *b, float *c) {
    register_kernel<<<grid_for(shape, 2 * TILE_SIZE), dim3(TILE_SIZE, TILE_SIZE)>>>(a, b, c, shape);
}
