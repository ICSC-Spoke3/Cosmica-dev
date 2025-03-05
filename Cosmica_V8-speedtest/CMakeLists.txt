cmake_minimum_required(VERSION 3.30)
set(CMAKE_CUDA_COMPILER /opt/miniforge3/envs/sdegno/bin/nvcc)
project(Cosmica LANGUAGES CXX CUDA)

include(FetchContent)

FetchContent_Declare(
        fkYAML
        GIT_REPOSITORY https://github.com/fktn-k/fkYAML.git
        GIT_TAG 6b36aac6a9907af2e921acf8ac12cc302b740977
)
FetchContent_MakeAvailable(fkYAML)

FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.15.1
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
        lyra
        GIT_REPOSITORY https://github.com/bfgroup/Lyra.git
        GIT_TAG e71849c1bc6dfb489464eb1a429e9be79d4f6745
)
FetchContent_MakeAvailable(lyra)

#-----------------------------------------------------------------------------
# Create executable from the CUDA source.
add_executable(Cosmica ${CMAKE_CURRENT_SOURCE_DIR}/kernel_test.cu)

#-----------------------------------------------------------------------------
# Set C++ and CUDA standards.
set_target_properties(Cosmica PROPERTIES
        CXX_STANDARD 20
        CUDA_STANDARD 20
)

#-----------------------------------------------------------------------------
# Find and enable OpenMP.
find_package(OpenMP REQUIRED)

# Set the include directory.
target_include_directories(Cosmica PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/headers)

# Define macros.
target_compile_definitions(Cosmica PRIVATE UNIFIED_COMPILE)

#-----------------------------------------------------------------------------
# Set the target CUDA architecture.
set_target_properties(Cosmica PROPERTIES CUDA_ARCHITECTURES 86)

#-----------------------------------------------------------------------------
# Add host compiler options (for C++). They will be forwarded to the CUDA compiler.
# Add CUDA-specific compile options.
set(HOST_FLAGS -fopenmp -Wall -Wextra -Werror)
set(CUDA_FLAGS -keep --generate-line-info --resource-usage --use_fast_math)
string(REPLACE ";" "," CUDA_HOST_FLAGS "${HOST_FLAGS}")

target_compile_options(Cosmica PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${HOST_FLAGS}>)
target_compile_options(Cosmica PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)
target_compile_options(Cosmica PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${CUDA_HOST_FLAGS}>)

#-----------------------------------------------------------------------------
# Link OpenMP libraries to resolve undefined references.
target_link_libraries(Cosmica PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(Cosmica PRIVATE fkYAML::fkYAML)
target_link_libraries(Cosmica PRIVATE spdlog::spdlog)
target_link_libraries(Cosmica PRIVATE bfg::lyra)

#-----------------------------------------------------------------------------
# Optionally, set the output directory for the executable.
set_target_properties(Cosmica PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/exefiles
)

#-----------------------------------------------------------------------------
# MOCK BUILD TARGET FOR CLION
add_executable(IgnoreThisTarget ${CMAKE_CURRENT_SOURCE_DIR}/kernel_test.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/DiffusionModel.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/EventSequence.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/GenComputation.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/GPUManage.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/HeliosphereModel.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/HeliosphericPropagation.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/IOConfiguration.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/HistoComputation.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/Histogram.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/LoadConfiguration.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/MagneticDrift.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/SDECoeffs.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sources/SolarWind.cu
)

set_target_properties(IgnoreThisTarget PROPERTIES
        CXX_STANDARD 20
        CUDA_STANDARD 20
)

target_include_directories(IgnoreThisTarget PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/headers)
target_compile_definitions(IgnoreThisTarget PRIVATE UNIFIED_COMPILE)
target_link_libraries(IgnoreThisTarget PRIVATE fkYAML::fkYAML)
target_link_libraries(IgnoreThisTarget PRIVATE spdlog::spdlog)
target_link_libraries(IgnoreThisTarget PRIVATE bfg::lyra)
