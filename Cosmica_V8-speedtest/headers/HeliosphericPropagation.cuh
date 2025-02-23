#ifndef HeliosphericPropagation
#define HeliosphericPropagation

// Propagation kernel cycle collecting the stocastic propagation computations of quasi-particle
__global__ void HeliosphericProp(unsigned, ThreadQuasiParticles_t, ThreadIndexes_t,
                                 SimulationParametrization_t, curandStatePhilox4_32_10_t *, float *);

#endif
