#include "HeliosphereModel.cuh"
#include "HelModVariableStructure.cuh"
//  Nose in Heliocentric Inertial (HCI o HGI) cartesian  
#define nose_x (-0.996f)
#define nose_y 0.03f
#define nose_z 0.088f
__host__ __device__ float Boundary(const float theta, const float phi, const float a, const float b) {
    /* Author: SDT - Mar 2018 -- updated Apr 2018 (formula from GLV) -- adapted for CUDA in Feb 2022
      * description: check if position is outside the spheroid that define the Heliosphere boundary.
      *              The spheroid is defined as two different spheres, one of radius b (tail distance) the second of radius a (nose distance).
      *              The transition between the two spheres is defined with the cosine^2 of the angle between the direction and the heliosphere Nose
      * Nose in ecliptic coord (degree) lat,lon  [ 5.3 254.7 ]
      * Nose in Heliocentric Inertial (HCI o HGI) cartesian  [ -0.996 0.03  0.088] HCI lat, long  [   5.121  178.269]
      */
    const float x = cos(phi) * sin(theta);
    const float y = sin(phi) * sin(theta);
    const float z = cos(theta);
    if (const float cosAlpha = x * nose_x + y * nose_y + z * nose_z; cosAlpha > 0) {
        return b - (b - a) * cosAlpha * cosAlpha;
    }
    return b;
}


__device__ int RadialZone(const unsigned int InitZone, const QuasiParticle_t &qp) {
    /* Author: SDT - Feb 2022
       * description: find in which zone the particle is.
       *              This function assume that, inside the Termination Shock, the heliosphere is divided in
       *              equally spatial Heliosphere.Nregions regions. outside TS there is 1 region.
       *              Each region is circular shaped, this means that last region is broader in tail direction.
       * return :  Number of zone between 0 and Heliosphere.Nregions-1 if inside TS
                   Heliosphere.Nregions if inside Heliosheat
                   -1 if outside Heliopause
       * // NOTE DEV rispetto alla versione HelMod si uniscono le funzioni RadialZone e WhereIAm
       */

    if (const auto [Rts_nose, Rhp_nose, Rts_tail, Rhp_tail] = Heliosphere.RadBoundary_effe[InitZone];
        qp.r < Boundary(qp.th, qp.phi, Rhp_nose, Rhp_tail)) {
        // inside Heliopause boundary
        if (qp.r >= Boundary(qp.th, qp.phi, Rts_nose, Rts_tail)) {
            // inside Heliosheat
            return Heliosphere.Nregions;
        }
        // inside Termination Shock Boundary
        if (qp.r < Rts_nose)
            return static_cast<int>(floorf(qp.r / Rts_nose * static_cast<float>(Heliosphere.Nregions)));
        return Heliosphere.Nregions - 1;
    }
    // outside heiosphere - Kill It
    return -1;
}
