#ifndef HeliosphereModel
#define HeliosphereModel

__host__ __device__ float Boundary(float, float, float, float);

__device__ int RadialZone(unsigned, const struct QuasiParticle_t & qp);

#endif
