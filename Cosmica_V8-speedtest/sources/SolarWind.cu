#include "GenComputation.cuh"
#include "HelModVariableStructure.cuh"

#include "HeliosphereModel.cuh"
#include "SolarWind.cuh"

////////////////////////////////////////////////////////////////
//..... Solar Wind Speed   .....................................
////////////////////////////////////////////////////////////////
__device__ float SolarWindSpeed(const unsigned char InitZone, const signed char HZone, const float r, const float th, const float phi) {
    /* Author: SDT - adapted for CUDA in Feb 2022
    * description: return the solar wind speed
    *
    * \param HZone Index of Heliosphere region
    * \param part  Pseudoparticle position
    */
    const float V0 = HZone < Heliosphere.Nregions ? LIM[HZone + InitZone].V0 : HS[InitZone].V0;
    // heliosheat (or near to)...............................
    if (HZone >= Heliosphere.Nregions - 1) {
        const float RtsDirection = Boundary(th, phi, Heliosphere.RadBoundary_effe[InitZone].Rts_nose,
                                            Heliosphere.RadBoundary_effe[InitZone].Rts_tail);
        if (r > RtsDirection - L_tl) {
            const float RtsRWDirection = Boundary(th, phi, Heliosphere.RadBoundary_real[InitZone].Rts_nose,
                                                  Heliosphere.RadBoundary_real[InitZone].Rts_tail);
            float DecreasFactor = SmoothTransition(1., 1. / s_tl, RtsDirection, L_tl, r);
            if (r > RtsDirection) {
                DecreasFactor *= powf(RtsRWDirection / (RtsRWDirection - RtsDirection + r), 2.);
            }
            return V0 * DecreasFactor;
        }
    }
    // inner Heliosphere .........................
    if (Heliosphere.IsHighActivityPeriod[InitZone]) {
        // high solar activity
        return V0;
    }
    // low solar activity
    float VswAngl = 1;
    if (Vhigh / V0 <= 2.) { VswAngl = Vhigh / V0 - 1.; }
    if (fabsf(cosf(th)) > VswAngl) {
        return Vhigh;
    }
    return V0 * (1 + fabsf(cosf(th)));
}

__device__ float DerivativeOfSolarWindSpeed_dtheta(const unsigned char InitZone, const signed char HZone, const float r, const float th,
                                                   const float phi) {
    /* Author: SDT - adapted for CUDA in Feb 2022
    * description: return the derivative of solar wind speed in d theta
    *
    * \param HZone Index of Heliosphere region
    * \param part  Pseudoparticle position
    */
    const float V0 = HZone < Heliosphere.Nregions ? LIM[HZone + InitZone].V0 : HS[InitZone].V0;
    // inner Heliosphere .........................
    if (Heliosphere.IsHighActivityPeriod[InitZone]) {
        // high solar activity
        return 0;
    }
    // low solar activity
    float VswAngl = 1;
    if (Vhigh / V0 <= 2.) { VswAngl = Vhigh / V0 - 1.; }
    if (fabsf(cosf(th)) > VswAngl) {
        return 0;
    }
    if (cosf(th) < 0) return V0 * sinf(th);
    if (cosf(th) == 0) return 0.;
    if (cosf(th) > 0) return -V0 * sinf(th);
    // heliosheat ...............................
    if (HZone >= Heliosphere.Nregions - 1) {
        const float RtsDirection = Boundary(th, phi, Heliosphere.RadBoundary_effe[InitZone].Rts_nose,
                                            Heliosphere.RadBoundary_effe[InitZone].Rts_tail);
        if (r > RtsDirection - L_tl) {
            return 0;
        }
    }
    return 0;
}
