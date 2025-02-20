#ifndef SDECoeffs
#define SDECoeffs
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"


__device__ DiffusionTensor_t DiffusionTensor_symmetric(const Index_t &, float, float, float,
                                                       float, PartDescription_t, float, const HeliosphereZoneProperties_t *LIM);

__device__ Tensor3D_t SquareRoot_DiffusionTerm(const Index_t &, DiffusionTensor_t, float, float, int *);

__device__ vect3D_t AdvectiveTerm(const Index_t &, const DiffusionTensor_t &, float, float,
                                  float, float, PartDescription_t, const HeliosphereZoneProperties_t *LIM);

__device__ float EnergyLoss(const Index_t &, float, float, float, float, const HeliosphereZoneProperties_t *LIM);

// __device__ float LossTerm(unsigned char,signed char, float, float, float, float, float);

#endif
