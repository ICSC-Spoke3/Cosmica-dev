#include "GenComputation.cuh"
#include "VariableStructure.cuh"

#include <math.h>           // c math library
#include <stdio.h>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions
#include <curand.h>         // CUDA random number host library
#include <curand_kernel.h>  // CUDA random number device library
#include <cuda_runtime.h>
#include <unistd.h>         // Supplies EXIT_FAILURE, EXIT_SUCCESS
#include <errno.h>          // Defines the external errno variable and all the values it can take on

////////////////////////////////////////////////////////////////
//..... GPU useful and safe function ...........................
//  
////////////////////////////////////////////////////////////////

__global__ void kernel_max(struct QuasiParticle_t *a, float *d, int Npart, int tpb)
{
  extern __shared__ float sdata[]; //"static" shared memory

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<Npart){
    sdata[tid] = (*a).R[i];
  }
  // if (blockIdx.x ==15 && threadIdx.x==199)
  // {
  //   printf("lll %u %.2f %.2f \n",i,a[i].part.Ek,sdata[tid]);
  // }
  __syncthreads();
  for(unsigned int s=tpb/2 ; s >= 1 ; s=s/2)
  {
    if(tid < s && i<Npart)
    {
      if(sdata[tid] < sdata[tid + s])
      {
        sdata[tid] = sdata[tid + s];
      }
    }
    __syncthreads();
  }
  if(tid == 0 ) 
  {
    d[blockIdx.x] = sdata[0];
  }
}

////////////////////////////////////////////////////////////////
//..... Random generator .......................................
//  
////////////////////////////////////////////////////////////////
__global__ void init_rdmgenerator(curandStatePhilox4_32_10_t *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}
