#ifndef HeliosphereModel
#define HeliosphereModel


__host__ __device__ float Boundary(float ,float ,float , float); 
/* * description: check if position is outside the spheroid that define the Heliosphere boundary. 
   */
   
#if TRIVIAL
  __device__ int Zone(float r, float th, float phi);

#else
   __device__ float RadialZone(int, float, float, float);
   /* * description: find in which zone the particle is.
      * return :  Number of zone between 0 and Heliosphere.Nregions-1 if inside TS
                  Heliosphere.Nregions if inside Heliosheat
                  -1 if outside Heliopause
      */
#endif
#endif