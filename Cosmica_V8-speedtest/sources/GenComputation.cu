#include "GenComputation.cuh"
#include "HelModVariableStructure.cuh"

#include <cmath>           // c math library

/**
 * @brief Safe sign function
 * @param num input number
 * @return 1 if num >= 0, -1 otherwise
 */
__device__ float safeSign(const float num) {
    if (num >= 0) return 1;
    return -1;
}

/**
 * @brief Smooth transition between  InitialVal to FinalVal centered at CenterOfTransition as function of x
 * if smoothness== 0 use a sharp transition
 * @param InitialVal initial value of the transition
 * @param FinalVal final value of the transition
 * @param CenterOfTransition center of the transition
 * @param smoothness smoothness of the transition (0 for step function)
 * @param x current value
 * @return If step function, returns InitialVal if x < CenterOfTransition, FinalVal otherwise
 * If smooth function, returns the value of the transition at x
 */
__device__ float SmoothTransition(const float InitialVal, const float FinalVal, const float CenterOfTransition,
                                  const float smoothness, const float x) {
    if (smoothness == 0) {
        if (x >= CenterOfTransition) return FinalVal;
        return InitialVal;
    }
    return (InitialVal + FinalVal) / 2.f - (InitialVal - FinalVal) / 2.f * tanhf(
               (x - CenterOfTransition) / smoothness);
}

// TODO: check unused
/**
 * @brief Returns the beta value
 * @param T kinetic energy
 * @param T0
 * @return the beta value
 */
__device__ float beta_(const float T, const float T0) {
    return sqrtf(T * (T + T0 + T0)) / (T + T0);
}

/**
 * @brief Returns the beta R value for a given rigidity
 * @param R rigidity
 * @param part particle description
 * @return the beta R value
 */
__device__ float beta_R(const float R, const PartDescription_t part) {
    return R / sqrtf(sq(R) + part.A * part.A / (part.Z * part.Z) * (part.T0 * part.T0));
}

// TODO: check unused
/**
 * @brief Return the rigidity for a given kinetic energy and particle description
 * @param T kinetic energy
 * @param part particle description
 * @return the rigidity
 */
__device__ __host__ float Rigidity(const float T, const PartDescription_t part) {
    return part.A / fabsf(part.Z) * sqrtf(T * (T + 2.f * part.T0));
}

/**
 * @brief Returns the energy for a given rigidity and particle description
 * @param R rigidity
 * @param part particle description
 * @return the energy
 */
__device__ __host__ float Energy(const float R, const PartDescription_t part) {
    return sqrtf(part.Z * part.Z / (part.A * part.A) * sq(R) + part.T0 * part.T0) - part.T0;
}
