cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(GrayScaleConvertor LANGUAGES CUDA CXX)
# 

include(CheckLanguage)
check_language(CUDA)

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW1_files main.cpp reference_calc.cpp compare.cpp HW1.cpp)

enable_language(CUDA)

#include(FindCUDA)
#set(CUDA_ARCH_LIST Auto CACHE STRING
#    "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
#compute capability versions (6.1, 7.0, etc) to generate code for. \
#Set to Auto for automatic detection (default)."
#)
#cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
#list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})

message(STATUS "CUDA ARCH = ${CUDA_ARCH_FLAGS}")

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

find_package(OpenCV REQUIRED)

# message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
# message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

add_executable(HW1 ${HW1_files} ${hdr} ${cu})
#add_compile_options(--device-debug --debug -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -ccbin g++ -m64)
#target_compile_options(
#    HW1 PRIVATE 
#     --device-debug --debug -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -ccbin g++ -m64)
# ${CUDA_ARCH_FLAGS}$<$<COMPILE_LANGUAGE:CUDA>:>
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --device-debug --debug -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -ccbin g++ -m64")

target_include_directories(
    HW1
    PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
    HW1
    ${CUDART_LIBRARY}
    ${OpenCV_LIBS}
)

