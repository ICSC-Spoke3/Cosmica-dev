#include "VariableStructure.cuh"

#include <curand_kernel.h>  // CUDA random number device library
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////
//..... GPU useful and safe function ...........................
//  
////////////////////////////////////////////////////////////////

__global__ void kernel_max(const ThreadQuasiParticles_t *a, float *d, const int Npart, const int tpb) {
    extern __shared__ float sdata[]; //"static" shared memory

    const unsigned int tid = threadIdx.x;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Npart) {
        sdata[tid] = a->R[i];
    }
    // if (blockIdx.x ==15 && threadIdx.x==199)
    // {
    //   printf("lll %u %.2f %.2f \n",i,a[i].part.Ek,sdata[tid]);
    // }
    __syncthreads();
    for (unsigned int s = tpb / 2; s >= 1; s = s / 2) {
        if (tid < s && i < Npart) {
            if (sdata[tid] < sdata[tid + s]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        d[blockIdx.x] = sdata[0];
    }
}

////////////////////////////////////////////////////////////////
//..... Random generator .......................................
//  
////////////////////////////////////////////////////////////////
__global__ void init_rdmgenerator(curandStatePhilox4_32_10_t *state, const unsigned long seed) {
    const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}
