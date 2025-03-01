#ifndef SDECoeffs
#define SDECoeffs
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"


__device__ DiffusionTensor_t DiffusionTensor_symmetric(const Index_t &, const QuasiParticle_t &qp, PartDescription_t,
                                                       float, SimulationParametrizations_t params);

__device__ Tensor3D_t SquareRoot_DiffusionTerm(const Index_t &, const QuasiParticle_t &, DiffusionTensor_t, int *);

__device__ vect3D_t AdvectiveTerm(const Index_t &, const QuasiParticle_t &, const DiffusionTensor_t &,
                                  PartDescription_t);

__device__ float EnergyLoss(const Index_t &, const QuasiParticle_t &);

// __device__ float LossTerm(unsigned char,signed char, float, float, float, float, float);

#endif
