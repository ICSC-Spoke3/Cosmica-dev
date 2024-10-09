#ifndef GenComputation
#define GenComputation

// ------------------------------------------
int ceil_int(int, int);
/* Ceil safe division for integer. Rounds x=a/b upward, returning the smallest integral value that is not less than x.
   */
int floor_int(int, int);
/* Floor safe division for integer. Rounds x=a/b downward, returning the biggest integral value that is less than x.
   */

__device__ float sign(float);
/* Return the signum of a the number
   */

__host__ __device__ float SmoothTransition(float, float, float, float, float);
/* Smooth transition between  InitialVal to FinalVal centered at CenterOfTransition as function of x if smoothness== 0 use a sharp transition
   */

__device__ float beta_(float, float);
/* Compute the beta value = v/c from kinetic energy
   */

__device__ float beta_R(float, struct PartDescription_t);
/* Compute the beta value = v/c from rigidity
   */

__device__ __host__ float Rigidity(float, struct PartDescription_t);
/* Convert Kinetic Energy in Rigidity
   */

__device__ __host__ float Energy(float R, struct PartDescription_t part);
/* Convert Rigidity in Kinetic Energy
   */

#endif


//#endif /* BCAC1E48_467C_43C1_9F78_615FEE9AAF65 */
