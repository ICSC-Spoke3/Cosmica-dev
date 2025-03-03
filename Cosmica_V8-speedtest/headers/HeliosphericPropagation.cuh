#ifndef HeliosphericPropagation
#define HeliosphericPropagation

__global__ void HeliosphericProp(ThreadQuasiParticles_t, ThreadIndexes_t, SimulationParametrizations_t,
                                 curandStatePhilox4_32_10_t *, float *);

#endif
