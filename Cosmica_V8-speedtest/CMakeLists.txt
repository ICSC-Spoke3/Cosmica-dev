cmake_minimum_required(VERSION 3.30)
set(CMAKE_CUDA_COMPILER /opt/miniforge3/envs/sdegno/bin/nvcc)
        # Project name and languages: CUDA and C++.
project(Cosmica LANGUAGES CUDA CXX)

# Optionally, locate OpenMP (if your host code uses OpenMP)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

if(OpenMP_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
endif()

# Set the CUDA standard.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# add_definitions(-DUNIFIED_COMPILE)

# Add the include directory (from the -I flag).
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/headers)

# Define the source files.
set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_test.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/DiffusionModel.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/GenComputation.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/GPUManage.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/HeliosphereModel.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/HeliosphericPropagation.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/HelModLoadConfiguration.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/HistoComputation.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/Histogram.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/LoadConfiguration.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/MagneticDrift.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/SDECoeffs.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/SolarWind.cu
)

# Create the executable target.
add_executable(Cosmica ${SOURCES})

# Specify the output directory (matching the NVCC -o option).
set_target_properties(Cosmica PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/exefiles
        # Enable relocatable device code; equivalent to "-rdc=true".
        CUDA_SEPARABLE_COMPILATION ON
)

# Add NVCC-specific compiler options.
target_compile_options(Cosmica
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
        $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler>
        $<$<COMPILE_LANGUAGE:CUDA>:-fopenmp>
)

# Instruct NVLink to use dynamic linking for libraries (fixes the libpthread error).
#target_link_options(Cosmica PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xnvlink;-Bdynamic>)

# (Optional) Link OpenMP libraries so that host code compiles with OpenMP support.
if(OpenMP_FOUND)
    target_link_libraries(Cosmica PRIVATE OpenMP::OpenMP_CXX)
endif()
