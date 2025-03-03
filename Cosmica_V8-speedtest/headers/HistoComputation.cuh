#ifndef HistoComputation
#define HistoComputation
#include <curand_kernel.h>

__global__ void init_rdmgenerator(curandStatePhilox4_32_10_t *, unsigned long);

__global__ void kernel_max(const struct QuasiParticle_t *, float *, int, int);

#endif
