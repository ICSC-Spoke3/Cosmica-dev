#ifndef HeliosphericPropagation
#define HeliosphericPropagation

// Propagation kernel cycle collecting the stocastic propagation computations of quasi-particle
__global__ void HeliosphericProp(unsigned int, float, float, float, QuasiParticle_t, ThreadIndexes,
                                 const HeliosphereZoneProperties_t *__restrict__,
                                 curandStatePhilox4_32_10_t *, float *);

#endif
