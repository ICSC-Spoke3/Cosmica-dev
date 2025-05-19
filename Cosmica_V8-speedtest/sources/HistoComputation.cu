#include "VariableStructure.cuh"

#include <cuda_runtime.h>

/**
 * @brief Kernel to compute the histogram of the energy of the particles;
 *
 * @param a ThreadQuasiParticles_t
 * @param d
 * @param Npart number of particles
 * @param tpb threads per block
 */
// TODO: check unused
__global__ void kernel_max(const ThreadQuasiParticles_t *a, float *d, const int Npart, const int tpb) {
    extern __shared__ float sdata[]; //"static" shared memory

    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Npart) {
        sdata[tid] = a->R[i];
    }
    __syncthreads();
    for (unsigned s = tpb / 2; s >= 1; s = s / 2) {
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

/**
 * @brief Generate random numbers for the particles for initialization of each thread
 * @param state curandStatePhilox4_32_10_t state
 * @param seed unsigned long seed for the random number generator
*  @note same seed for all threads but different random sequences along the random array for each thread
 */
__global__ void init_rdmgenerator(curandStatePhilox4_32_10_t *state, const unsigned long seed) {
    const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}
