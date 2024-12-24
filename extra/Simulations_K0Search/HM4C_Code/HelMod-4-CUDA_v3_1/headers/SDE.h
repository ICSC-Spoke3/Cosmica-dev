#ifndef SDE
#define SDE

// ------------------------------------------
__device__ Tensor3D_t SquareRoot_DiffusionTerm(signed short ,Tensor3D_t , qvect_t , int & ); 
/* * ========== Decomposition of Diffusion Tensor ====== 
     solve the square root of diffusion tensor in heliocentric spherical coordinates
   */
__device__ vect3D_t AdvectiveTerm(unsigned short,signed short ,DiffusionTensor_t , qvect_t,PartDescription_t );
/* * Advective term of SDE
   */
__device__ float EnergyLoss(unsigned short,signed short, particle_t ); 
/* */
__device__ float LossTerm(unsigned short,signed short, particle_t ); 
// ------------------------------------------

#endif
