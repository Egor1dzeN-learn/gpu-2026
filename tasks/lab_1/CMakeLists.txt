add_executable(lab_1 main.cpp cuda/cuda_softmax.cu)


target_link_libraries(lab_1 PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(lab_1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h
    HINTS "${CUDNN_ROOT}" ENV CUDNN_ROOT ENV CUDA_PATH
    PATH_SUFFIXES include)
find_library(CUDNN_LIBRARY NAMES cudnn cudnn64_9 cudnn64_8
    HINTS "${CUDNN_ROOT}" ENV CUDNN_ROOT ENV CUDA_PATH
    PATH_SUFFIXES lib/x64 lib)

if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    target_sources(lab_1 PRIVATE cudnn/cudnn_softmax.cpp)
    target_compile_definitions(lab_1 PRIVATE LAB1_USE_CUDNN)
    target_include_directories(lab_1 PRIVATE
        "${CUDNN_INCLUDE_DIR}" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(lab_1 PRIVATE "${CUDNN_LIBRARY}")
    message(STATUS "lab_1: cuDNN enabled")
elseif(CUDNN_ROOT OR DEFINED ENV{CUDNN_ROOT})
    message(FATAL_ERROR "cuDNN headers or library not found. Check CUDNN_ROOT.")
else()
    message(STATUS "lab_1: cuDNN not found. Set CUDNN_ROOT to enable it.")
endif()
