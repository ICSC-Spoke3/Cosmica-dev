#include "GenComputation.cuh"
#include "HelModVariableStructure.cuh"

#include "HeliosphereModel.cuh"
#include "SolarWind.cuh"

////////////////////////////////////////////////////////////////
//..... Solar Wind Speed   .....................................
////////////////////////////////////////////////////////////////
__device__ float SolarWindSpeed(const unsigned int InitZone, const signed int HZone, const float r, const float th,
                                const float phi) {
    /* Author: SDT - adapted for CUDA in Feb 2022
    * description: return the solar wind speed
    *
    * \param HZone Index of Heliosphere region
    * \param part  Pseudoparticle position
    */
    const float V0 = HZone < Heliosphere.Nregions ? LIM[HZone + InitZone].V0 : HS[InitZone].V0;


    // heliosheat (or near to)...............................
    if (const float RtsDirection = Boundary(th, phi, Heliosphere.RadBoundary_effe[InitZone].Rts_nose,
                                            Heliosphere.RadBoundary_effe[InitZone].Rts_tail);
        HZone >= Heliosphere.Nregions - 1 && r > RtsDirection - L_tl) {
        const float RtsRWDirection = Boundary(th, phi, Heliosphere.RadBoundary_real[InitZone].Rts_nose,
                                              Heliosphere.RadBoundary_real[InitZone].Rts_tail);
        float DecreasFactor = SmoothTransition(1., 1. / s_tl, RtsDirection, L_tl, r);
        if (r > RtsDirection) {
            DecreasFactor *= sq(RtsRWDirection / (RtsRWDirection - RtsDirection + r));
        }
        return V0 * DecreasFactor;
    }

    // inner Heliosphere .........................
    if (Heliosphere.IsHighActivityPeriod[InitZone]) {
        // high solar activity
        return V0;
    }

    return min(Vhigh, V0 * (1 + fabsf(cosf(th))));
}

__device__ float DerivativeOfSolarWindSpeed_dtheta(const unsigned int InitZone, const signed int HZone, const float r,
                                                   const float th, const float phi) {
    /* Author: SDT - adapted for CUDA in Feb 2022
    * description: return the derivative of solar wind speed in d theta
    *
    * \param HZone Index of Heliosphere region
    * \param part  Pseudoparticle position
    */
    const float V0 = HZone < Heliosphere.Nregions ? LIM[HZone + InitZone].V0 : HS[InitZone].V0;

    // heliosheat ...............................
    // inner Heliosphere .........................
    if (const float RtsDirection = Boundary(th, phi, Heliosphere.RadBoundary_effe[InitZone].Rts_nose,
                                            Heliosphere.RadBoundary_effe[InitZone].Rts_tail);
        (HZone >= Heliosphere.Nregions - 1 && r > RtsDirection - L_tl) ||
        Heliosphere.IsHighActivityPeriod[InitZone] ||
        V0 * (1 + fabsf(cosf(th))) > Vhigh
    ) {
        return 0;
    }

    return -sign(cosf(th)) * V0 * sinf(th);
}
