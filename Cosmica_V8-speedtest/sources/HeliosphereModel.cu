#include "HeliosphereModel.cuh"
#include "HelModVariableStructure.cuh"
//  Nose in Heliocentric Inertial (HCI o HGI) cartesian  
#define nose_x (-0.996f)
#define nose_y 0.03f
#define nose_z 0.088f

/**
 * @brief  This function calculates the boundary of the heliosphere.
 * Check if position is outside the spheroid that define the Heliosphere boundary.
 * The spheroid is defined as two different spheres, one of radius b (tail distance) the second of radius a (nose distance).
 * The transition between the two spheres is defined with the cosine^2 of the angle between the direction and the heliosphere Nose
 * @param  theta: polar angle
 * @param  phi: azimuthal angle
 * @param  a: nose distance
 * @param  b: tail distance
 * @return boundary distance
 * @note  Nose in ecliptic coord (degree) lat,lon  [ 5.3 254.7 ], Nose in Heliocentric Inertial (HCI o HGI) cartesian  [ -0.996 0.03  0.088] HCI lat, long  [   5.121  178.269]
 */
__host__ __device__ float Boundary(const float theta, const float phi, const float a, const float b) {
    const float x = cos(phi) * sin(theta);
    const float y = sin(phi) * sin(theta);
    const float z = cos(theta);
    if (const float cosAlpha = x * nose_x + y * nose_y + z * nose_z; cosAlpha > 0) {
        return b - (b - a) * cosAlpha * cosAlpha;
    }
    return b;
}

/**
 * @brief  This function RadialZone calculates the zone in which the particle is.
 * The function assumes that, inside the Termination Shock, the heliosphere is divided in
 * equally spatial Heliosphere.Nregions regions. outside TS there is 1 region.
 * Each region is circular shaped, this means that last region is broader in tail direction.
 * @param  InitZone: initial zone
 * @param  qp: QuasiParticle_t struct with particle information
 * @return Number of zone between 0 and Heliosphere.Nregions-1 if inside TS or Heliosheat, Heliosphere.Nregions if inside Heliosheat, -1 if outside Heliopause
 */
__device__ int RadialZone(const unsigned InitZone, const QuasiParticle_t &qp) {
    if (const auto [Rts_nose, Rhp_nose, Rts_tail, Rhp_tail] = Constants.RadBoundary_effe[InitZone];
        qp.r < Boundary(qp.th, qp.phi, Rhp_nose, Rhp_tail)) {
        // inside Heliopause boundary
        if (qp.r >= Boundary(qp.th, qp.phi, Rts_nose, Rts_tail)) {
            // inside Heliosheat
            return Constants.Nregions;
        }
        // inside Termination Shock Boundary
        if (qp.r < Rts_nose)
            return static_cast<int>(floorf(qp.r / Rts_nose * static_cast<float>(Constants.Nregions)));
        return Constants.Nregions - 1;
    }
    // outside heiosphere - Kill It
    return -1;
}
