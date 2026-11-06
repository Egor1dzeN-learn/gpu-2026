#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "naiveCPU/naive_cpu.h"
#include "omp/omp_softmax.h"
#include "omp_simd/omp_simd_softmax.h"
#include "cuda/cuda_softmax.h"

namespace {

    constexpr int kWarmupRuns = 3;
    constexpr int kMeasuredRuns = 10;
    constexpr unsigned kSeed = 42;

    struct Size {
        size_t rows;
        size_t cols;
    };

    const std::vector<Size> kSizesFixedRows = {
            {1024, 128},
            {1024, 256},
            {1024, 512},
            {1024, 1024},
            {1024, 2048},
            {1024, 4096},
            {1024, 8192},
    };

    const std::vector<Size> kSizesFixedCols = {
            {128,  1024},
            {256,  1024},
            {512,  1024},
            {1024, 1024},
            {2048, 1024},
            {4096, 1024},
            {8192, 1024},
    };

    constexpr int kPhysicalCores = 6;

    // small, medium, large
    const std::vector<Size> kSizesScaling = {
            {1024, 128},
            {1024, 1024},
            {1024, 8192},
    };

    std::vector<float> make_input(size_t rows, size_t cols, unsigned seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> data(rows * cols);
        for (float &x: data) {
            x = dist(gen);
        }
        return data;
    }

    float median(std::vector<float> values) {
        std::sort(values.begin(), values.end());
        const size_t n = values.size();
        if (n % 2 == 1) {
            return values[n / 2];
        }
        return 0.5f * (values[n / 2 - 1] + values[n / 2]);
    }

    struct ReferenceError {
        double max_abs = 0.0;
        double max_rel = 0.0;
    };

    ReferenceError compare_to_reference(const std::vector<float> &out, const std::vector<float> &ref) {
        ReferenceError err;
        for (size_t i = 0; i < out.size(); ++i) {
            const double abs_err = std::fabs(static_cast<double>(out[i]) - ref[i]);
            err.max_abs = std::max(err.max_abs, abs_err);
            err.max_rel = std::max(err.max_rel, abs_err / std::fabs(ref[i]));
        }
        return err;
    }

    void benchmark(const char *series, const char *impl_name, AbstractSoftmax &impl,
                   AbstractSoftmax &reference, const std::vector<Size> &sizes) {
        for (const Size &size: sizes) {
            std::vector<float> input = make_input(size.rows, size.cols, kSeed);
            const std::vector<float> expected = reference.run(input.data(), size.cols, size.rows).results;

            for (int r = 0; r < kWarmupRuns; ++r) {
                impl.run(input.data(), size.cols, size.rows);
            }

            std::vector<float> times;
            std::vector<float> kernel_times;
            times.reserve(kMeasuredRuns);
            kernel_times.reserve(kMeasuredRuns);
            SoftmaxResult last;
            for (int r = 0; r < kMeasuredRuns; ++r) {
                last = impl.run(input.data(), size.cols, size.rows);
                times.push_back(last.execution_time);
                kernel_times.push_back(last.kernel_time);
            }

            if (last.results.size() != size.rows * size.cols) {
                std::fprintf(stderr, "%s %zux%zu: wrong output size %zu\n",
                             impl_name, size.rows, size.cols, last.results.size());
                continue;
            }

            const ReferenceError ref_err = compare_to_reference(last.results, expected);

            std::printf("%s,%s,%zu,%zu,%.4f,%.4f,%.3e,%.3e\n",
                        series, impl_name, size.rows, size.cols,
                        median(times), median(kernel_times),
                        ref_err.max_abs, ref_err.max_rel);
        }
    }

}

int main() {
    NaiveCPU naive_cpu;
    OmpSoftmax ompSoftmax1(1);
    OmpSoftmax ompSoftmax2(2);
    OmpSoftmax ompSoftmax4(4);
    OmpSoftmax ompSoftmaxMax;
    OmpSIMDSoftmax ompSimdSoftmax;
    CudaSoftmax cudaSoftmax;
    std::printf("series,impl,M,N,median_ms,median_kernel_ms,max_abs_err,max_rel_err\n");
    benchmark("fixed_M", "naive_cpu", naive_cpu, naive_cpu, kSizesFixedRows);
    benchmark("fixed_N", "naive_cpu", naive_cpu, naive_cpu, kSizesFixedCols);
    benchmark("fixed_M", "omp1", ompSoftmax1, naive_cpu, kSizesFixedRows);
    benchmark("fixed_N", "omp1", ompSoftmax1, naive_cpu, kSizesFixedCols);
    benchmark("fixed_M", "omp2", ompSoftmax2, naive_cpu, kSizesFixedRows);
    benchmark("fixed_N", "omp2", ompSoftmax2, naive_cpu, kSizesFixedCols);
    benchmark("fixed_M", "omp4", ompSoftmax4, naive_cpu, kSizesFixedRows);
    benchmark("fixed_N", "omp4", ompSoftmax4, naive_cpu, kSizesFixedCols);
    benchmark("fixed_M", "ompMax", ompSoftmaxMax, naive_cpu, kSizesFixedRows);
    benchmark("fixed_N", "ompMax", ompSoftmaxMax, naive_cpu, kSizesFixedCols);
    benchmark("fixed_M", "ompSIMD", ompSimdSoftmax, naive_cpu, kSizesFixedRows);
    benchmark("fixed_N", "ompSIMD", ompSimdSoftmax, naive_cpu, kSizesFixedCols);
    benchmark("fixed_M", "cuda", cudaSoftmax, naive_cpu, kSizesFixedRows);
    benchmark("fixed_N", "cuda", cudaSoftmax, naive_cpu, kSizesFixedCols);

    for (int threads = 1; threads <= kPhysicalCores; ++threads) {
        OmpSoftmax omp(threads);
        const std::string name = "omp" + std::to_string(threads);
        benchmark("scaling", name.c_str(), omp, naive_cpu, kSizesScaling);
    }

    return 0;
}
