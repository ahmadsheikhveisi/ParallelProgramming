cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(GrayScaleConvertor LANGUAGES CUDA CXX)
# 

include(CheckLanguage)
check_language(CUDA)

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)

    
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


add_executable(Reduce ${hdr} ${cu})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --device-debug --debug -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -ccbin g++ -m64")

target_include_directories(
    Reduce
    PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(
    Reduce
    ${CUDART_LIBRARY}
)




