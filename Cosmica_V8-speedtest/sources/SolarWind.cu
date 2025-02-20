#include "GenComputation.cuh"
#include "HelModVariableStructure.cuh"

#include "HeliosphereModel.cuh"
#include "SolarWind.cuh"

/**
 * @brief Calculate the solar wind speed at a given position in the heliosphere.
 *
 * @param index
 * @param r Radial distance
 * @param th th
 * @param phi phi
 * @param LIM
 * @return Solar wind speed
 */
__device__ float SolarWindSpeed(const Index_t &index, const float r,
                                const float th, const float phi, const HeliosphereZoneProperties_t *LIM) {
    const float V0 = index.radial < Heliosphere.Nregions ? LIM[index.combined()].V0 : HS[index.period].V0;


    // heliosheat (or near to)...............................
    if (const float RtsDirection = Boundary(th, phi, Heliosphere.RadBoundary_effe[index.period].Rts_nose,
                                            Heliosphere.RadBoundary_effe[index.period].Rts_tail);
        index.radial >= Heliosphere.Nregions - 1 && r > RtsDirection - L_tl) {
        const float RtsRWDirection = Boundary(th, phi, Heliosphere.RadBoundary_real[index.period].Rts_nose,
                                              Heliosphere.RadBoundary_real[index.period].Rts_tail);
        float DecreasFactor = SmoothTransition(1.f, 1.f / s_tl, RtsDirection, L_tl, r);
        if (r > RtsDirection) {
            DecreasFactor *= sq(RtsRWDirection / (RtsRWDirection - RtsDirection + r));
        }
        return V0 * DecreasFactor;
    }

    // inner Heliosphere .........................
    if (Heliosphere.IsHighActivityPeriod[index.period]) {
        // high solar activity
        return V0;
    }

    return min(Vhigh, V0 * (1 + fabsf(cosf(th))));
}

/**
 * @brief Derivative of solar wind speed in d theta
 *
 * @param index
 * @param r Radial distance
 * @param th th
 * @param phi phi
 * @param LIM
 * @return Derivative of solar wind speed in d theta
 */
__device__ float DerivativeOfSolarWindSpeed_dtheta(const Index_t &index,
                                                   const float r, const float th, const float phi, const HeliosphereZoneProperties_t *LIM) {
    const float V0 = index.radial < Heliosphere.Nregions ? LIM[index.combined()].V0 : HS[index.period].V0;

    // heliosheat ...............................
    // inner Heliosphere .........................
    if (const float RtsDirection = Boundary(th, phi, Heliosphere.RadBoundary_effe[index.period].Rts_nose,
                                            Heliosphere.RadBoundary_effe[index.period].Rts_tail);
        (index.radial >= Heliosphere.Nregions - 1 && r > RtsDirection - L_tl) ||
        Heliosphere.IsHighActivityPeriod[index.period] ||
        V0 * (1 + fabsf(cosf(th))) > Vhigh
    ) {
        return 0;
    }

    return -sign(cosf(th)) * V0 * sinf(th);
}
