#ifndef HeliosphericPropagation
#define HeliosphericPropagation

// Propagation kernel cycle collecting the stocastic propagation computations of quasi-particle
__global__ void HeliosphericProp(int, struct QuasiParticle_t, int*, struct PartDescription_t, curandStatePhilox4_32_10_t*, float*);

#endif