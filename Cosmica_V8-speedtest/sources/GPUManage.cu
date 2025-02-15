#include "GPUManage.cuh"
#include "VariableStructure.cuh"
#include "GenComputation.cuh"

#include <cstdio>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions
#include <curand_kernel.h>  // CUDA random number device library
#include <cuda_runtime.h>   // Device code management by providing implicit initialization, context management, and module management



#include <HeliosphericPropagation.cuh>


#include <iostream>


#include <unistd.h>         // Supplies EXIT_FAILURE, EXIT_SUCCESS

void HandleError(const cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

constexpr unsigned long hash(const std::string_view &str) {
    unsigned long hash = 0;
    for (const auto &e: str) hash = hash * 131 + e;
    return hash;
}

consteval unsigned long operator""_(const char *str, const size_t len) {
    return hash(std::string_view(str, len));
}

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

LaunchParam_t RoundNpart(const int NPart, cudaDeviceProp GPUprop, const bool verbose, const int WpB, const int svars) {
    LaunchParam_t launch_param;

    // Computation of the number of blocks, warp per blocks, threads per block and shared memory bits
    launch_param.Npart = ceil_int(NPart, GPUprop.warpSize) * GPUprop.warpSize;
    int WarpPerBlock = WpB <= 0 ? BestWarpPerBlock(GPUprop.name, verbose) : WpB;
    launch_param.threads = WarpPerBlock * GPUprop.warpSize;
    launch_param.blocks = ceil_int(launch_param.Npart, launch_param.threads);
    // Use a minimum of 2 blocks per Single Multiprocessor (cuda prescription)
    if (launch_param.blocks < 2) launch_param.blocks = 2;

    launch_param.smem = static_cast<int>(svars * launch_param.threads * sizeof(float));

    if (launch_param.threads > GPUprop.maxThreadsPerBlock || launch_param.blocks > GPUprop.maxGridSize[0]) {
        fprintf(stderr, "------- propagation Kernel -----------------\n");
        fprintf(stderr, "ERROR:: Number of Threads per block or number of blocks not allowed for this device\n");
        fprintf(stderr, "        Number of Threads per Block setted %d - max allowed %d\n", launch_param.threads,
                GPUprop.maxThreadsPerBlock);
        fprintf(stderr, "        Number of Blocks setted %d - max allowed %d\n", launch_param.blocks,
                GPUprop.maxGridSize[0]);
        exit(EXIT_FAILURE);
    }

// #define EXPERIMENTAL_GRID
#ifdef EXPERIMENTAL_GRID
    int gridSize, minGridSize, blockSize = 32;
    int maxActiveBlocks = 0;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, HeliosphericProp, blockSize, 0);
    // gridSize = GPUprop.multiProcessorCount * maxActiveBlocks;
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, HeliosphericProp, 0, 65536 / 86);
    // blockSize = (blockSize + GPUprop.warpSize - 1) / GPUprop.warpSize * GPUprop.warpSize;
    gridSize = (NPart + blockSize - 1) / blockSize;

    launch_param.blocks = gridSize;
    launch_param.threads = blockSize;
    launch_param.smem = static_cast<int>(svars * launch_param.threads * sizeof(float));
#endif

    if (verbose) {
        printf("------- propagation Kernel -----------------\n");
        printf("-- Number of particle which will be simulated: %d\n", launch_param.Npart);
        printf("-- Number of Warp in a Block       : %d \n", WarpPerBlock);
        printf("-- Number of blocks                : %d \n", launch_param.blocks);
        printf("-- Number of threadsPerBlock       : %d \n", launch_param.threads);
        printf("-- Shared Memory                   : %d \n", launch_param.smem);
        printf("-- \n\n");
    }

    return launch_param;
}

cudaDeviceProp *DeviceInfo(const int N_GPU_count, const bool verbose) {
    const auto infos = new cudaDeviceProp[N_GPU_count];

    // Retrive the GPU properties (info) for each available GPU
    for (int i = 0; i < N_GPU_count; i++)
        cudaGetDeviceProperties(&infos[i], i);


    // Print all the GPU info useful for debugging
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
