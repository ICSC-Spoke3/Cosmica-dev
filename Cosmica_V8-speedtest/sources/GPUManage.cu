#include <iostream>
#include <cstdio>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions
#include <cuda_runtime.h>   // Device code management by providing implicit initialization, context management, and module management


#include "GPUManage.cuh"
#include "VariableStructure.cuh"
#include "GenComputation.cuh"

/**
 * @brief HandleError
 * @param err cudaError_t error
 * @param file the file where the error occurred
 * @param line the line where the error occurred
 */
void HandleError(const cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief hash function to convert a string into a hash
 * @param str the string to be hashed
 * @return the hash of the string
 */
constexpr unsigned long hash(const std::string_view &str) {
    unsigned long hash = 0;
    for (const auto &e: str) hash = hash * 131 + e;
    return hash;
}

/**
 * @brief operator""_ to convert a string into a hash
 * @param str the string to be hashed
 * @param len the length of the string
 * @return the hash of the string
 */
consteval unsigned long operator""_(const char *str, const size_t len) {
    return hash(std::string_view(str, len));
}

/**
 * @brief Define the best value of NWarpPerBlock for a given GPU (name)
 * @param name the name of the GPU
 * @param verbose the verbosity of the output
 * @return the best warp per block
 */
int BestWarpPerBlock(char name[], const int verbose) {
    int BestWarpPerBlock = 8;

    switch (hash(name)) {
        case "NVIDIA A30"_:
        case "NVIDIA A40"_:
            BestWarpPerBlock = 2;
            break;
        case "NVIDIA A100"_:
            BestWarpPerBlock = 16;
            break;
        default:
            std::cerr << "WARNING: best value not known, used default warp per block = 8 for " << name << std::endl;
    }

    if (verbose) {
        printf("----- Simulation infos -----\n");
        printf("-- For board %s we execute the code using NWarpPerBlock=%d\n", name, BestWarpPerBlock);
    }

    return BestWarpPerBlock;
}

/**
 * @brief Round the number of particle to be simulated based on the GPU capability, returning a struct with threads
 * and blocks count together with shared memory bytes.
 * The calculations are done based on the cuda occupancy calculator output to maximize the usage of the GPUs.
 * @param NPart the number of particles
 * @param GPUprop the GPU properties
 * @param verbose the verbosity of the output
 * @param WpB the warp per block
 * @return the launch parameters
 */
LaunchParam_t RoundNpart(const unsigned NPart, cudaDeviceProp GPUprop, const bool verbose, const int WpB) {
    LaunchParam_t launch_param;

    // Computation of the number of blocks, warp per blocks, threads per block and shared memory bits
    int WarpPerBlock = WpB <= 0 ? BestWarpPerBlock(GPUprop.name, verbose) : WpB;
    launch_param.threads = WarpPerBlock * GPUprop.warpSize;
    launch_param.blocks = ceil_int_div(NPart, launch_param.threads);
    // Use a minimum of 2 blocks per Single Multiprocessor (cuda prescription)
    if (launch_param.blocks < 2) launch_param.blocks = 2;

    if (launch_param.threads > static_cast<unsigned>(GPUprop.maxThreadsPerBlock) ||
        launch_param.blocks > static_cast<unsigned>(GPUprop.maxGridSize[0])) {
        fprintf(stderr, "------- propagation Kernel -----------------\n");
        fprintf(stderr, "ERROR:: Number of Threads per block or number of blocks not allowed for this device\n");
        fprintf(stderr, "        Number of Threads per Block setted %d - max allowed %d\n", launch_param.threads,
                GPUprop.maxThreadsPerBlock);
        fprintf(stderr, "        Number of Blocks setted %d - max allowed %d\n", launch_param.blocks,
                GPUprop.maxGridSize[0]);
        exit(EXIT_FAILURE);
    }

#define EXPERIMENTAL_GRID
#ifdef EXPERIMENTAL_GRID
    // int gridSize, minGridSize, blockSize = 32;
    // int maxActiveBlocks = 0;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, HeliosphericProp, blockSize, 0);
    // gridSize = GPUprop.multiProcessorCount * maxActiveBlocks;
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, HeliosphericProp, 0, 65536 / 86);
    // blockSize = (blockSize + GPUprop.warpSize - 1) / GPUprop.warpSize * GPUprop.warpSize;
    // gridSize = (NPart + blockSize - 1) / blockSize;

    // launch_param.threads = (768 + GPUprop.warpSize - 1) / GPUprop.warpSize * GPUprop.warpSize;
    launch_param.threads = 48;
    launch_param.blocks = (NPart + launch_param.threads - 1) / launch_param.threads;
#endif

    if (verbose) {
        printf("------- propagation Kernel -----------------\n");
        printf("-- Number of particle which will be simulated: %d\n", NPart);
        printf("-- Number of Warp in a Block       : %d \n", WarpPerBlock);
        printf("-- Number of blocks                : %d \n", launch_param.blocks);
        printf("-- Number of threadsPerBlock       : %d \n", launch_param.threads);
        printf("-- \n\n");
    }

    return launch_param;
}

/**
 * @brief Retrieve the GPU properties (info) for the selected GPU and print a summary of useful infos for debugging,
 * with the verbose option.
 * @param N_GPU_count the number of GPU
 * @param verbose the verbosity of the output
 * @return the GPU properties
 */
cudaDeviceProp *DeviceInfo(const int N_GPU_count, const bool verbose) {
    const auto infos = new cudaDeviceProp[N_GPU_count];

    for (int i = 0; i < N_GPU_count; i++)
        cudaGetDeviceProperties(&infos[i], i);

    if (verbose) {
        printf("----- GPU infos -----\n");
        printf("There are %d CUDA enabled devices \n", N_GPU_count);

        for (int i = 0; i < N_GPU_count; i++) {
            printf("--   --- General Information for device %d ---\n", i);
            printf("-- Name:  %s\n", infos[i].name);
            printf("-- Compute capability:  %d.%d\n", infos[i].major, infos[i].minor);
            printf("-- Clock rate:  %d\n", infos[i].clockRate);
            printf("-- Device copy overlap:  ");
            if (infos[i].deviceOverlap)
                printf("Enabled\n");
            else
                printf("Disabled\n");
            printf("-- Kernel execution timeout :  ");
            if (infos[i].kernelExecTimeoutEnabled)
                printf("Enabled\n");
            else
                printf("Disabled\n");

            printf("--    --- Memory Information for device %d ---\n", i);
            printf("-- Total global mem:  %ld\n", infos[i].totalGlobalMem);
            printf("-- Total constant Mem:  %ld\n", infos[i].totalConstMem);
            printf("-- Max mem pitch:  %ld\n", infos[i].memPitch);
            printf("-- Texture Alignment:  %ld\n", infos[i].textureAlignment);

            printf("--    --- MP Information for device %d ---\n", i);
            printf("-- Multiprocessor count:  %d\n",
                   infos[i].multiProcessorCount);
            printf("-- Shared mem per mp:  %ld\n", infos[i].sharedMemPerBlock);
            printf("-- Registers per mp:  %d\n", infos[i].regsPerBlock);
            printf("-- Threads in warp:  %d\n", infos[i].warpSize);
            printf("-- Max threads per block:  %d\n",
                   infos[i].maxThreadsPerBlock);
            printf("-- Max thread dimensions:  (%d, %d, %d)\n",
                   infos[i].maxThreadsDim[0], infos[i].maxThreadsDim[1],
                   infos[i].maxThreadsDim[2]);
            printf("-- Max grid dimensions:  (%d, %d, %d)\n",
                   infos[i].maxGridSize[0], infos[i].maxGridSize[1],
                   infos[i].maxGridSize[2]);
            printf("-- \n\n");
        }
    }

    return infos;
}
