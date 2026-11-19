#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <string_view>
#include <vector>
#include <omp.h>

#include "naiveCPU/naive_cpu.h"
#include "omp/omp_matmul.h"
#include "omp_simd/omp_simd_matmul.h"
#include "cuda/cuda_matmul.h"
#include "cublas/cublas_matmul.h"

namespace {

constexpr int kWarmupRuns = 1;
constexpr int kMeasuredRuns = 3;
constexpr int kPhysicalCores = 6;
constexpr unsigned kSeed = 42;
constexpr std::array kSizes{64, 128, 256, 512, 1024, 2048, 4096, 8192};
constexpr std::array kSizesScaling{128, 512, 2048};

struct BenchmarkData {
    MatmulShape shape;
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> expected;
    std::vector<float> output;
};

BenchmarkData make_input(int size) {
    const size_t count = static_cast<size_t>(size) * size;
    BenchmarkData data{{size, size, size}, std::vector<float>(count), std::vector<float>(count),
                       std::vector<float>(count), std::vector<float>(count)};
    std::mt19937 generator(kSeed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (float &value : data.a) {
        value = distribution(generator);
    }
    for (float &value : data.b) {
        value = distribution(generator);
    }
    return data;
}

template <typename T>
T median(std::array<T, kMeasuredRuns> &values, int count) {
    std::sort(values.begin(), values.begin() + count);
    if (count % 2 == 0) {
        return (values[count / 2 - 1] + values[count / 2]) / 2;
    }
    return values[count / 2];
}

struct ReferenceError {
    double max_abs = 0.0;
    double max_rel = 0.0;
};

ReferenceError compare_to_reference(const std::vector<float> &out, const std::vector<float> &reference) {
    ReferenceError error;
    for (size_t i = 0; i < out.size(); ++i) {
        if (!std::isfinite(out[i])) {
            return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
        }
        const double expected = reference[i];
        const double difference = std::abs(out[i] - expected);
        const double denominator = std::max(std::abs(expected), 1e-7);
        error.max_abs = std::max(error.max_abs, difference);
        error.max_rel = std::max(error.max_rel, difference / denominator);
    }
    return error;
}

void benchmark(const char *series, const char *impl_name, int threads, AbstractMatmul &&impl,
               BenchmarkData &data, int repetitions, bool is_reference = false) {
    auto &output = is_reference ? data.expected : data.output;
    for (int i = 0; i < kWarmupRuns; ++i) {
        impl.run(data.a.data(), data.b.data(), output.data());
    }

    std::array<double, kMeasuredRuns> times{};
    std::array<float, kMeasuredRuns> kernel_times{};
    for (int i = 0; i < repetitions; ++i) {
        const MatmulTime time = impl.run(data.a.data(), data.b.data(), output.data());
        times[i] = time.total_ms;
        kernel_times[i] = time.kernel_ms;
    }

    const ReferenceError error = is_reference ? ReferenceError{} : compare_to_reference(output, data.expected);
    std::printf("%s,%s,%d,%d,%d,%d,%.6f,%.6f,%.6e,%.6e\n",
                series, impl_name, threads, data.shape.m, data.shape.k, data.shape.n,
                median(times, repetitions), median(kernel_times, repetitions), error.max_abs, error.max_rel);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char **argv) {
    int max_size = kSizes.back();
    if (argc == 2) {
        const std::string_view argument(argv[1]);
        const auto end = argument.data() + argument.size();
        const auto parsed = std::from_chars(argument.data(), end, max_size);
        if (parsed.ec != std::errc{} || parsed.ptr != end || max_size < kSizes.front() || max_size > kSizes.back()) {
            max_size = 0;
        }
    }
    if (argc > 2 || max_size == 0) {
        std::fprintf(stderr, "Usage: lab_2 [max_size from 64 to 8192]\n");
        return 1;
    }

    omp_set_dynamic(0);
    std::printf("series,impl,threads,M,K,N,median_ms,median_kernel_ms,max_abs_err,max_rel_err\n");
    for (int size : kSizes) {
        if (size > max_size) {
            break;
        }

        BenchmarkData data = make_input(size);
        const MatmulShape shape = data.shape;
        const int cpu_repetitions = size >= 4096 ? 1 : kMeasuredRuns;

        // CPU
        benchmark("matmul", "naive_cpu", 1, NaiveMatmul(shape), data, cpu_repetitions, true);
        benchmark("matmul", "omp", 1, OmpMatmul(shape, 1), data, cpu_repetitions);
        benchmark("matmul", "omp", 2, OmpMatmul(shape, 2), data, cpu_repetitions);
        benchmark("matmul", "omp", 4, OmpMatmul(shape, 4), data, cpu_repetitions);
        benchmark("matmul", "omp_simd", kPhysicalCores, OmpSimdMatmul(shape, kPhysicalCores), data, cpu_repetitions);

        // GPU
        benchmark("matmul", "cuda_naive", 0, CudaNaiveMatmul(shape), data, kMeasuredRuns);
        benchmark("matmul", "cuda_shared", 0, CudaSharedMatmul(shape), data, kMeasuredRuns);
        benchmark("matmul", "cuda_registers", 0, CudaRegisterMatmul(shape), data, kMeasuredRuns);
        benchmark("matmul", "cublas", 0, CublasMatmul(shape), data, kMeasuredRuns);

        // OpenMP scaling
        if (std::find(kSizesScaling.begin(), kSizesScaling.end(), size) != kSizesScaling.end()) {
            for (int threads = 1; threads <= kPhysicalCores; ++threads) {
                benchmark("scaling", "omp", threads, OmpMatmul(shape, threads), data, kMeasuredRuns);
            }
        }
    }
    return 0;
}
