#ifndef GenComputation
#define GenComputation
// #define sq(x) ((x)*(x))
// #define cb(x) ((x)*(x)*(x))

// ------------------------------------------
template <std::integral T>
__forceinline__ constexpr T ceil_int_div(const T a, const T b) {
   return (a + (b - 1)) / b;
}

template <std::integral T>
__forceinline__ constexpr T floor_int_div(const T a, const T b) {
   return a / b - (a % b && a < 0 != b < 0);
}

__device__ float safeSign(float);

/* Return the signum of a the number
   */

__device__ __forceinline__ float sign(const float val) {
    return static_cast<float>((0.f < val) - (val < 0.f));
}

__device__ __host__ __forceinline__ float sq(const float val) {
    return val * val;
}

__device__ __host__ __forceinline__ float clamp(const float f, const float a, const float b)
{
   return fmaxf(a, fminf(f, b));
}

__device__ __forceinline__ float atomicMax(float *address, const float val)
{
   int ret = __float_as_int(*address);
   while(val > __int_as_float(ret))
   {
      if(const int old = ret; (ret = atomicCAS(reinterpret_cast<int *>(address), old, __float_as_int(val))) == old)
         break;
   }
   return __int_as_float(ret);
}

__host__ __device__ float SmoothTransition(float, float, float, float, float);

/* Smooth transition between  InitialVal to FinalVal centered at CenterOfTransition as function of x if smoothness== 0 use a sharp transition
   */

__device__ float beta_(float, float);

/* Compute the beta value = v/c from kinetic energy
   */

__device__ float beta_R(float, struct PartDescription_t);

/* Compute the beta value = v/c from rigidity
   */

__device__ __host__ float Rigidity(float, PartDescription_t);

/* Convert Kinetic Energy in Rigidity
   */

__device__ __host__ float Energy(float, PartDescription_t);

/* Convert Rigidity in Kinetic Energy
   */

#endif


//#endif /* BCAC1E48_467C_43C1_9F78_615FEE9AAF65 */
