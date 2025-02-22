#ifndef HeliosphericPropagation
#define HeliosphericPropagation

// Propagation kernel cycle collecting the stocastic propagation computations of quasi-particle
__global__ void HeliosphericProp(unsigned int, float, float, float, ThreadQuasiParticles_t, ThreadIndexes_t,
                                 const HeliosphereZoneProperties_t *__restrict__,
                                 curandStatePhilox4_32_10_t *, float *);

#endif
