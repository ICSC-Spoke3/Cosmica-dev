#ifndef HeliosphereModel
#define HeliosphereModel


__host__ __device__ float Boundary(float, float, float, float);

/* * description: check if position is outside the spheroid that define the Heliosphere boundary.
   */

__device__ int RadialZone(unsigned, const struct QuasiParticle_t & qp);

/* * description: find in which zone the particle is.
   * return :  Number of zone between 0 and Heliosphere.Nregions-1 if inside TS
               Heliosphere.Nregions if inside Heliosheat
               -1 if outside Heliopause
   */

#endif
