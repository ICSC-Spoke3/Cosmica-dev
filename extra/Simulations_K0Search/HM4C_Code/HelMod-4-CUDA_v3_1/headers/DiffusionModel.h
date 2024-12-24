#ifndef DiffusionModel
#define DiffusionModel



void RescaleToEffectiveHeliosphere(HeliosphereBoundRadius_t &,vect3D_t &);
float K0Fit_ssn(float , float , float , float *);
float K0Fit_NMC(float , float *);
float K0CorrFactor(int, int, int, float);
float3 EvalK0(bool , int , int, int , float , float ,float , unsigned char );
float g_low(int , int , float );
float rconst(int , int , float );
__device__ float3 Diffusion_Tensor_In_HMF_Frame(unsigned short, signed short , float , float , float , float , float , float3 &);
__device__ float  Diffusion_Coeff_heliosheat(signed short , qvect_t  , float , float , float &);
#endif
