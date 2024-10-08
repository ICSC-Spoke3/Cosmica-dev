#ifndef SDECoeffs
#define SDECoeffs

// Convective-diffusive tensor of quasi-particle SDE propagation
__device__ struct DiffusionTensor_t trivial_DiffusionTensor_symmetric();

// Square root decomposition of the convective-diffusive tensor
__device__ struct Tensor3D_t trivial_SquareRoot_DiffusionTensor(float, float, struct PartDescription_t);

// Advective-drift vector tensor of quasi-particle SDE propagation
__device__ struct vect3D_t trivial_AdvectiveTerm(struct Tensor3D_t, float, float);

// Energy loss term of quasi-particle SDE propagation
__device__ float trivial_EnergyLoss(float, float, float, struct PartDescription_t);

// AlphaPath weight of the histom counts for exit energy
__device__ float trivial_LossTerm(float, float, float, struct PartDescription_t);



__device__ struct DiffusionTensor_t DiffusionTensor_symmetric(unsigned char, signed char, float, float, float, float, struct PartDescription_t, float);

__device__ struct Tensor3D_t SquareRoot_DiffusionTerm(signed char, struct DiffusionTensor_t, float, float, int*);

__device__ struct vect3D_t AdvectiveTerm(unsigned char,signed char, struct DiffusionTensor_t, float, float, float, float, struct PartDescription_t);

__device__ float EnergyLoss(unsigned char, signed char, float, float, float, float, float);

__device__ float LossTerm(unsigned char,signed char, float, float, float, float, float);

#endif