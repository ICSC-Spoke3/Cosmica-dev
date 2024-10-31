#include "GPUManage.cuh"
#include "VariableStructure.cuh"
#include "GenComputation.cuh"

#include <math.h>           // c math library
#include <stdio.h>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions
#include <curand.h>         // CUDA random number host library
#include <curand_kernel.h>  // CUDA random number device library
#include <cuda_runtime.h>   // Device code management by providing implicit initialization, context management, and module management
#include <unistd.h>         // Supplies EXIT_FAILURE, EXIT_SUCCESS
#include <errno.h>          // Defines the external errno variable and all the values it can take on
  
void HandleError(cudaError_t err, const char* file, int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit( EXIT_FAILURE );
    }
}

int BestWarpPerBlock (char name[], int verbose) {
    int BestWarpPerBlock = 8;
  
    if      (strstr(name, "NVIDIA A30"))  {BestWarpPerBlock=2;} // cuda cap. 8.0 : 1,2,4,8,16,32 
    else if (strstr(name, "NVIDIA A40"))  {BestWarpPerBlock=2;} // cuda cap. 8.6 : 1,2,3,4,6,8,12,16,24 
    else if (strstr(name, "NVIDIA A100")) {BestWarpPerBlock=16;} // cuda cap. 8.0 : 1,2,4,8,16,32 
    /* more else if clauses */
  
    else { 
        fprintf (stderr,"WARNING: %s best value not knows, used default warp per block = 8\n", name);
    }
  
    if (verbose) {
        printf( "----- Simulation infos -----\n" );
        printf("-- For board %s we execute the code using NWarpPerBlock=%d\n", name, BestWarpPerBlock );
    }
  
    return BestWarpPerBlock;
}
  
struct LaunchParam_t RoundNpart (int NPart, cudaDeviceProp GPUprop, bool verbose, int WpB) {
    struct LaunchParam_t launch_param;

    // Computation of the number of blocks, warp per blocks, threads per block and shared memory bits
    launch_param.Npart = ceil_int(NPart, (GPUprop.warpSize))*(GPUprop.warpSize);
    int WarpPerBlock;
    if (WpB<=0) WarpPerBlock = BestWarpPerBlock(GPUprop.name, verbose);
    else WarpPerBlock = WpB;
    launch_param.threads = (int)WarpPerBlock*GPUprop.warpSize;
    launch_param.blocks = (int)ceil_int(launch_param.Npart, launch_param.threads);
    // Use a minimum of 2 blocks per Single Multiprocessor (cuda prescription)
    if (launch_param.blocks < 2) {
        launch_param.blocks = 2;
    }
    launch_param.smem = (int)(24*launch_param.threads*sizeof(float));

    if (launch_param.threads>GPUprop.maxThreadsPerBlock || launch_param.blocks>GPUprop.maxGridSize[0]) {
        fprintf(stderr,"------- propagation Kernel -----------------\n");
        fprintf(stderr,"ERROR:: Number of Threads per block or number of blocks not allowed for this device\n");
        fprintf(stderr,"        Number of Threads per Block setted %d - max allowed %d\n", launch_param.threads, GPUprop.maxThreadsPerBlock);
        fprintf(stderr,"        Number of Blocks setted %d - max allowed %d\n", launch_param.blocks, GPUprop.maxGridSize[0]);
        exit(EXIT_FAILURE);
      }
  
      if (verbose) {
  
          printf("------- propagation Kernel -----------------\n");
          printf("-- Number of particle which will be simulated: %d\n", launch_param.Npart);
          printf("-- Number of Warp in a Block       : %d \n", WarpPerBlock);
          printf("-- Number of blocks                : %d \n", launch_param.blocks);
          printf("-- Number of threadsPerBlock       : %d \n", launch_param.threads);
          printf( "-- \n\n" );
  
      }

    return launch_param;
}

cudaDeviceProp* DeviceInfo (int N_GPU_count, bool verbose) {

    cudaDeviceProp* infos = (cudaDeviceProp*)malloc(N_GPU_count*sizeof(cudaDeviceProp));

    // Retrive the GPU properties (info) for each available GPU
    for (int i=0; i< N_GPU_count; i++) {
        cudaGetDeviceProperties(&(infos[i]), i);
    }

    // Print all the GPU info useful for debugging
    if (verbose) {
        printf( "----- GPU infos -----\n" );
        printf( "There are %d CUDA enabled devices \n",N_GPU_count );
        
        for (int i=0; i< N_GPU_count; i++) {
            printf( "--   --- General Information for device %d ---\n", i );
            printf( "-- Name:  %s\n", infos[i].name );
            printf( "-- Compute capability:  %d.%d\n", infos[i].major, infos[i].minor );
            printf( "-- Clock rate:  %d\n", infos[i].clockRate );
            printf( "-- Device copy overlap:  " );
            if (infos[i].deviceOverlap)
                printf( "Enabled\n" );
            else
                printf( "Disabled\n");
            printf( "-- Kernel execution timeout :  " );
            if (infos[i].kernelExecTimeoutEnabled)
                printf( "Enabled\n" );
            else
                printf( "Disabled\n" );

            printf( "--    --- Memory Information for device %d ---\n", i );
            printf( "-- Total global mem:  %ld\n", infos[i].totalGlobalMem );
            printf( "-- Total constant Mem:  %ld\n", infos[i].totalConstMem );
            printf( "-- Max mem pitch:  %ld\n", infos[i].memPitch );
            printf( "-- Texture Alignment:  %ld\n", infos[i].textureAlignment );

            printf( "--    --- MP Information for device %d ---\n", i );
            printf( "-- Multiprocessor count:  %d\n",
                        infos[i].multiProcessorCount );
            printf( "-- Shared mem per mp:  %ld\n", infos[i].sharedMemPerBlock );
            printf( "-- Registers per mp:  %d\n", infos[i].regsPerBlock );
            printf( "-- Threads in warp:  %d\n", infos[i].warpSize );
            printf( "-- Max threads per block:  %d\n",
                        infos[i].maxThreadsPerBlock );
            printf( "-- Max thread dimensions:  (%d, %d, %d)\n",
                        infos[i].maxThreadsDim[0], infos[i].maxThreadsDim[1],
                        infos[i].maxThreadsDim[2] );
            printf( "-- Max grid dimensions:  (%d, %d, %d)\n",
                        infos[i].maxGridSize[0], infos[i].maxGridSize[1],
                        infos[i].maxGridSize[2] );
            printf( "-- \n\n" );
        }
    }

    return infos;
}