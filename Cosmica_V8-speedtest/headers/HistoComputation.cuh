#ifndef HistoComputation
#define HistoComputation

__global__ void init_rdmgenerator(curandStatePhilox4_32_10_t *, unsigned long);

/* Generate initializing random numbers for each thread (same seed for all threads but different random sequences
 * along the random array for each thread)
   */
__global__ void kernel_max(const struct QuasiParticle_t *, float *, int, int);

/* Find maximum d in array a (note that d is an array of lenght nblock)
   */
// ------------------------------------------

#endif
