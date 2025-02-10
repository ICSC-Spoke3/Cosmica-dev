#ifndef GPUManage
#define GPUManage

void HandleError(cudaError_t, const char *, int);

/* Manage error in GPU
   */
struct LaunchParam_t RoundNpart(int, cudaDeviceProp, bool, int, int);

/* Round the number of particle to be simulated based on the GPU capability, returning a struct with threads
   and blocks count together with shared memory bytes.
   The calculations are done based on the cuda occupancy calculator output to maximize the usage of the GPUs.
   */

int BestNWarpPerBlock(char, bool);

/* Define the best value of NWarpPerBlock for a given GPU (name) at the website
   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities in table 15 one may
   found the value of Maximum number of resident warps per SM (MSWSM) for each Compute Capability. The number
   of WarpPerBlock should be a divisor of MSWSM to allow a full occupancy of SM. The best value of WarpPerBlock
   should be found with a performance study, together with Cuda Occupancy Calculator.
   */

cudaDeviceProp *DeviceInfo(int, bool);

/* Retrive the GPU properties (info) for the selected GPU and print a summary of useful infos for debugging,
   with the verbose option.
   */

#endif
