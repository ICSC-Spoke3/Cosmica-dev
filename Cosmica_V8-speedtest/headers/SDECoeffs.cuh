#ifndef SDECoeffs
#define SDECoeffs
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"

// Convective-diffusive tensor of quasi-particle SDE propagation
__device__ DiffusionTensor_t trivial_DiffusionTensor_symmetric(int);

// Square root decomposition of the convective-diffusive tensor
__device__ Tensor3D_t trivial_SquareRoot_DiffusionTensor(DiffusionTensor_t);

// Advective-drift vector tensor of quasi-particle SDE propagation
__device__ vect3D_t trivial_AdvectiveTerm(DiffusionTensor_t);

// Energy loss term of quasi-particle SDE propagation
__device__ float trivial_EnergyLoss();


__device__ DiffusionTensor_t DiffusionTensor_symmetric(unsigned int, signed int, float, float, float, float,
                                                       PartDescription_t, float);

__device__ Tensor3D_t SquareRoot_DiffusionTerm(signed int, DiffusionTensor_t, float, float, int *);

__device__ vect3D_t AdvectiveTerm(unsigned int, signed int, const DiffusionTensor_t &, float, float, float,
                                  float, PartDescription_t);

__device__ float EnergyLoss(unsigned int, signed int, float, float, float, float);

// __device__ float LossTerm(unsigned char,signed char, float, float, float, float, float);

#endif
