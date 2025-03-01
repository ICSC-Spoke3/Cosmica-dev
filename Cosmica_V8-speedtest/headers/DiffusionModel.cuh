#ifndef DiffusionModel
#define DiffusionModel
#include <tuple>
#include "VariableStructure.cuh"


void RescaleToEffectiveHeliosphere(HeliosphereBoundRadius_t &, InitialPositions_t &, unsigned);

float K0Fit_ssn(float, float, float, float *);

float K0Fit_NMC(float, float *);

float K0CorrFactor(float, float, float, float);

std::tuple<float, float, float> EvalK0(bool, int, int, int, float, float, float, unsigned char);

float g_low(int, int, float);

float rconst(int, int, float);

__device__ float3
Diffusion_Tensor_In_HMF_Frame(const Index_t &, const QuasiParticle_t &qp, float, float, float3 &,
                              SimulationParametrizations_t);

__device__ float Diffusion_Coeff_heliosheat(const Index_t &, const QuasiParticle_t &, float, float &);
#endif
