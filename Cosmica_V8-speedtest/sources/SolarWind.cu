#include "GenComputation.cuh"
#include "HelModVariableStructure.cuh"

#include "HeliosphereModel.cuh"
#include "SolarWind.cuh"

/**
 * @brief Calculate the solar wind speed at a given position in the heliosphere.
 *
 * @param index
 * @param qp the quasi-particle
 * @return Solar wind speed at the given position;
 * @note If the quasi-particle is outside the heliosphere, the solar wind speed is simply V0.
 */
__device__ float SolarWindSpeed(const Index_t &index, const QuasiParticle_t &qp) {
    const float V0 = index.radial < Constants.Nregions
                         ? Constants.heliosphere_properties[index.combined()].V0
                         : Constants.heliosheat_properties[index.period].V0;

    if (const float RtsDirection = Boundary(qp.th, qp.phi, Constants.RadBoundary_effe[index.period].Rts_nose,
                                            Constants.RadBoundary_effe[index.period].Rts_tail);
        index.radial >= Constants.Nregions - 1 && qp.r > RtsDirection - L_tl) {
        const float RtsRWDirection = Boundary(qp.th, qp.phi, Constants.RadBoundary_real[index.period].Rts_nose,
                                              Constants.RadBoundary_real[index.period].Rts_tail);
        float DecreasFactor = SmoothTransition(1.f, 1.f / s_tl, RtsDirection, L_tl, qp.r);
        if (qp.r > RtsDirection) {
            DecreasFactor *= sq(RtsRWDirection / (RtsRWDirection - RtsDirection + qp.r));
        }
        return V0 * DecreasFactor;
    }

    if (Constants.IsHighActivityPeriod[index.period]) {
        return V0;
    }
    return min(Vhigh, V0 * (1 + fabsf(cosf(qp.th))));
}

/**
 * @brief Derivative of solar wind speed with respect to theta.
 *
 * @param index
 * @param qp the quasi-particle
 * @return Derivative of solar wind speed in d theta
 */
__device__ float DerivativeOfSolarWindSpeed_dtheta(const Index_t &index, const QuasiParticle_t &qp) {
    const float V0 = index.radial < Constants.Nregions
                         ? Constants.heliosphere_properties[index.combined()].V0
                         : Constants.heliosheat_properties[index.period].V0;

    if (const float RtsDirection = Boundary(qp.th, qp.phi, Constants.RadBoundary_effe[index.period].Rts_nose,
                                            Constants.RadBoundary_effe[index.period].Rts_tail);
        (index.radial >= Constants.Nregions - 1 && qp.r > RtsDirection - L_tl) || Constants.IsHighActivityPeriod[index.
            period] || V0 * (1 + fabsf(cosf(qp.th))) > Vhigh
    ) {
        return 0;
    }

    return -sign(cosf(qp.th)) * V0 * sinf(qp.th);
}
