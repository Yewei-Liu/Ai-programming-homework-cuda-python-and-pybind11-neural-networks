cmake_minimum_required(VERSION 3.12)
project(cmake_example)

add_subdirectory(third_party/pybind11)

find_package(Python3 REQUIRED COMPONENTS Development)

enable_language(CUDA)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

include_directories(${Python3_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

pybind11_add_module(mytensor src/tmp.cu)

target_link_libraries(mytensor PRIVATE ${CUDA_LIBRARIES} cublas ${Python3_LIBRARIES})

set_target_properties(mytensor PROPERTIES LINKER_LANGUAGE CUDA)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
