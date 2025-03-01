#include "GenComputation.cuh"
#include "HelModVariableStructure.cuh"

#include <cmath>           // c math library

////////////////////////////////////////////////////////////////
//..... GEneric useful and safe function .......................
////////////////////////////////////////////////////////////////

__device__ float safeSign(const float num) {
    if (num >= 0) return 1;
    return -1;
}

__device__ float SmoothTransition(const float InitialVal, const float FinalVal, const float CenterOfTransition,
                                  const float smoothness,
                                  const float x) {
    if (smoothness == 0) {
        if (x >= CenterOfTransition) return FinalVal;
        return InitialVal;
    }
    return (InitialVal + FinalVal) / 2.f - (InitialVal - FinalVal) / 2.f * tanhf(
               (x - CenterOfTransition) / smoothness);
}

__device__ float beta_(const float T, const float T0) {
    return sqrtf(T * (T + T0 + T0)) / (T + T0);
}

__device__ float beta_R(const float R, const PartDescription_t part) {
    // float T = Energy(R, part);
    // return beta_(T, part.T0);
    return R / sqrtf(sq(R) + part.A * part.A / (part.Z * part.Z) * (part.T0 * part.T0));
}

__device__ __host__ float Rigidity(const float T, const PartDescription_t part) {
    return part.A / fabsf(part.Z) * sqrtf(T * (T + 2.f * part.T0));
}

__device__ __host__ float Energy(const float R, const PartDescription_t part) {
    return sqrtf(part.Z * part.Z / (part.A * part.A) * sq(R) + part.T0 * part.T0) - part.T0;
}
