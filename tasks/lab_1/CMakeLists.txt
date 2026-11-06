add_executable(lab_1 main.cpp cuda/cuda_softmax.cu)


target_link_libraries(lab_1 PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(lab_1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
