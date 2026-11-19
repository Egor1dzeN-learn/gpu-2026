find_package(CUDAToolkit REQUIRED)

set(cpu_sources
    naiveCPU/naive_cpu.cpp
    omp/omp_matmul.cpp
    omp_simd/omp_simd_matmul.cpp)

add_executable(lab_2 main.cpp ${cpu_sources} cuda/cuda_matmul.cu cublas/cublas_matmul.cpp)
target_include_directories(lab_2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lab_2 PRIVATE OpenMP::OpenMP_CXX CUDA::cublas)
if(MSVC)
    set_source_files_properties(${cpu_sources} PROPERTIES COMPILE_OPTIONS /arch:AVX2)
else()
    set_source_files_properties(${cpu_sources} PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()
