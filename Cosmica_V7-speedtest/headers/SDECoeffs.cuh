#ifndef SDECoeffs
#define SDECoeffs

// Convective-diffusive tensor of quasi-particle SDE propagation
__device__ struct DiffusionTensor_t trivial_DiffusionTensor_symmetric(int);

// Square root decomposition of the convective-diffusive tensor
__device__ struct Tensor3D_t trivial_SquareRoot_DiffusionTensor(struct DiffusionTensor_t);

// Advective-drift vector tensor of quasi-particle SDE propagation
__device__ struct vect3D_t trivial_AdvectiveTerm(struct DiffusionTensor_t);

// Energy loss term of quasi-particle SDE propagation
__device__ float eval_trivial_EnergyLoss();

__device__ float eval_Bth(float, float, int);

__device__ float eval_Bph(float, float, float, float);

__device__ float eval_HMF_Mag(float, float, bool, float, float);

__device__ float eval_sqrtBR2BT2(float, float);

__device__ float eval_sinPsi(float, float, float);

__device__ float eval_cosPsi(float, float, float);

__device__ float eval_sinZeta(float, float, float);

__device__ float eval_cosZeta(float, float, float);

__device__ float eval_dBth_dr(float, float);

__device__ float eval_DsinPsi_dr(float, float, bool, float, float, float, float, float, float, float);

__device__ float eval_DsinPsi_dtheta(float, float, bool, float, float, float, float, float, float, float, float);

__device__ float eval_dBph_dr(float, float, float, float);

__device__ float eval_dBth_dth(float, float, float);

__device__ float eval_dBph_dth(float, float, float, float, float, float);

__device__ float eval_dBMag_dth(float, float, float, float, float, float, float, float);

__device__ float eval_dBMag_dr(float, float, float, float, float, float, float);

__device__ float eval_dsqrtBR2BT2_dr(float, float, float, float, float);

__device__ float eval_dsqrtBR2BT2_dth(float, float, float, float, float);

__device__ float eval_DcosPsi_dr(float, float, float, float, float, float, float, float);

__device__ float eval_DcosPsi_dtheta(float, float, float, float, float, float, float, float);

__device__ float eval_DsinZeta_dr(float, float, float, float, float);

__device__ float eval_DsinZeta_dtheta(float, float, float, float, float);

__device__ float eval_DcosZeta_dr(float, float, float, float, float);

__device__ float eval_DcosZeta_dtheta(float, float, float, float, float);

__device__ DiffusionTensor_t DiffusionTensor_symmetric(unsigned char, signed char, float, float, float, float, struct PartDescription_t, float);

__device__ Tensor3D_t SquareRoot_DiffusionTerm(signed char, float, float, float, float, float, float, float, float);

__device__ float AdvectiveTerm_radius(float, unsigned char,signed char, float, float, float, float, float, float, float, float, struct PartDescription_t);

__device__ float AdvectiveTerm_theta(float, unsigned char,signed char, float, float, float, float, float, float, float, float, struct PartDescription_t);

__device__ float AdvectiveTerm_phi(float, unsigned char,signed char, float, float, float, float, float, float, float, struct PartDescription_t);

__device__ float EnergyLoss(unsigned char, signed char, float, float, float, float);

// __device__ float LossTerm(unsigned char,signed char, float, float, float, float, float);

#endif