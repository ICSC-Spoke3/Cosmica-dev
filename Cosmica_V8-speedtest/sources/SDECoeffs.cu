#include <cmath>
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"
#include "GenComputation.cuh"
#include "SolarWind.cuh"
#include "MagneticDrift.cuh"
#include "DiffusionModel.cuh"
#include "SDECoeffs.cuh"


/**
 * @brief Calculate the diffusion tensor in the HMF frame
 * @param index
 * @param qp the quasi-particle
 * @param pt
 * @param GaussRndNumber
 * @param params
 * @return the diffusion tensor in the HMF frame
 * @note The implemented model is the Parker’s IMF with Jokipii and Kota (1989) modification in polar regions.
 * @note HMF Parameterization: In the equatorial region, we use Parker’s Interplanetary Magnetic Field (IMF) (B_Par) based on the
 *   parametrization by Hattingh and Burger (1995).
 * @note In the polar regions, we apply a modified IMF (B_Pol) that incorporates a latitudinal component to
 *   account for large-scale fluctuations, which dominate at high heliolatitudes, as suggested by
 *   Jokipii and Kota (1989).
 * @note The boundary between equatorial and polar regions is defined by the constants 'PolarZone' and
 *   'CosPolarZone' (see globals.h).
 * @note Since the theta component of the HMF is zero in the equatorial region, we implemented two separate
 *   cases to avoid unnecessary calculations.
 */
// TODO: complete documentation
__device__ DiffusionTensor_t DiffusionTensor_symmetric(const Index_t &index, const QuasiParticle_t &qp,
                                                       const PartDescription_t pt, const float GaussRndNumber,
                                                       const SimulationParametrizations_t params) {
    DiffusionTensor_t KK;
    if (index.radial < Constants.Nregions) {
        const bool IsPolarRegion = fabsf(cosf(qp.th)) > CosPolarZone;
        const float SignAsun = Constants.heliosphere_properties[index.combined()].Asun > 0 ? +1. : -1.;

        // @formatter:off
        float3 dK_dr;
        const auto [Kpar, Kperp1, Kperp2] =
                Diffusion_Tensor_In_HMF_Frame(index, qp, beta_R(qp.R, pt), GaussRndNumber, dK_dr, params);

        const float PolSign = qp.th - Pi / 2.f > 0 ? -1.f : +1.f;
        const float DelDirac = qp.th == Pi / 2.f ? 1.f : 0.f;
        const float V_SW = SolarWindSpeed(index, qp);

        const float Br = PolSign;
        const float Bth = IsPolarRegion ? qp.r * delta_m / (rhelio * sinf(qp.th)) : 0.f;
        const float Bph = -PolSign * (Omega * (qp.r - rhelio) * sinf(qp.th) / V_SW);

        const float dBr_dr = -2.f * PolSign / qp.r;
        const float dBr_dth = -2.f * DelDirac;

        const float dBth_dr = IsPolarRegion ? -delta_m / (rhelio * sinf(qp.th)) : 0.f;
        const float dBth_dth = IsPolarRegion ? qp.r * delta_m / (rhelio * sinf(qp.th) * sinf(qp.th)) * -cosf(qp.th) : 0.f;

        const float dBph_dr = PolSign * (qp.r - 2.f * rhelio) * (Omega * sinf(qp.th)) / (qp.r * V_SW);
        const float dBph_dth = -(qp.r - rhelio) * Omega * (-PolSign * (cosf(qp.th) * V_SW - sinf(qp.th) * DerivativeOfSolarWindSpeed_dtheta(index, qp)) + 2.f * sinf(qp.th) * V_SW * DelDirac) / (V_SW * V_SW);

        const float HMF_Mag2 = 1 + Bth * Bth + Bph * Bph;
        const float HMF_Mag = sqrtf(HMF_Mag2);
        const float dBMag_dr = (Br * dBr_dr + Bth * dBth_dr + Bph * dBph_dr) / HMF_Mag;
        const float dBMag_dth = (Br * dBr_dth + Bth * dBth_dth + Bph * dBph_dth) / HMF_Mag;

        const float sqrtBR2BT2 = sqrtf(sq(Br) + sq(Bth));
        const float dsqrtBR2BT2_dr = (Br * dBr_dr + Bth * dBth_dr) / sqrtBR2BT2;
        const float dsqrtBR2BT2_dth = (Br * dBr_dth + Bth * dBth_dth) / sqrtBR2BT2;


        const float sinPsi = SignAsun * (-Bph / HMF_Mag);
        const float cosPsi = sqrtBR2BT2 / HMF_Mag;
        const float sinZeta = SignAsun * (Bth / sqrtBR2BT2);
        const float cosZeta = SignAsun * (Br / sqrtBR2BT2);

        const float DsinPsi_dr = -SignAsun * (dBph_dr * HMF_Mag - Bph * dBMag_dr) / HMF_Mag2;
        const float DsinPsi_dtheta = -SignAsun * (dBph_dth * HMF_Mag - Bph * dBMag_dth) / HMF_Mag2;


        const float DcosPsi_dr = Bph * (-Br * Br * dBph_dr + Bph * Br * dBr_dr + Bth * (-Bth * dBph_dr + Bph * dBth_dr)) / (sqrtBR2BT2 * HMF_Mag2 * HMF_Mag);
        const float DcosPsi_dtheta = Bph * (-Br * Br * dBph_dth + Bph * Br * dBr_dth + Bth * (-Bth * dBph_dth + Bph * dBth_dth)) / (sqrtBR2BT2 * HMF_Mag2 * HMF_Mag);

        const float DsinZeta_dr = SignAsun * (dBth_dr * sqrtBR2BT2 - Bth * dsqrtBR2BT2_dr) / (sqrtBR2BT2 * sqrtBR2BT2);
        const float DsinZeta_dtheta = SignAsun * (dBth_dth * sqrtBR2BT2 - Bth * dsqrtBR2BT2_dth) / (sqrtBR2BT2 * sqrtBR2BT2);
        const float DcosZeta_dr = SignAsun * (dBr_dr * sqrtBR2BT2 - Br * dsqrtBR2BT2_dr) / (sqrtBR2BT2 * sqrtBR2BT2);
        const float DcosZeta_dtheta = SignAsun * (dBr_dth * sqrtBR2BT2 - Br * dsqrtBR2BT2_dth) / (sqrtBR2BT2 * sqrtBR2BT2);

        KK.rr = Kperp1 * sq(sinZeta) + sq(cosZeta) * (Kpar * sq(cosPsi) + Kperp2 * sq(sinPsi));
        KK.tt = Kperp1 * sq(cosZeta) + sq(sinZeta) * (Kpar * sq(cosPsi) + Kperp2 * sq(sinPsi));
        KK.pp = Kpar * sq(sinPsi) + Kperp2 * sq(cosPsi);
        KK.tr = sinZeta * cosZeta * (Kpar * sq(cosPsi) + Kperp2 * sq(sinPsi) - Kperp1);
        KK.pr = -(Kpar - Kperp2) * sinPsi * cosPsi * cosZeta;
        KK.pt = -(Kpar - Kperp2) * sinPsi * cosPsi * sinZeta;

        KK.DKrr_dr = 2.f * cosZeta * (sq(cosPsi) * Kpar + Kperp2 * sq(sinPsi)) * DcosZeta_dr + sinZeta * sinZeta * dK_dr.y + sq(cosZeta) * (2.f * cosPsi * Kpar * DcosPsi_dr + sq(cosPsi) * dK_dr.x + sinPsi * (sinPsi * dK_dr.z + 2.f * Kperp2 * DsinPsi_dr)) + 2.f * Kperp1 * sinZeta * DsinZeta_dr;
        KK.DKtt_dt = 2.f * cosZeta * Kperp1 * DcosZeta_dtheta + sq(sinZeta) * (2.f * cosPsi * Kpar * DcosPsi_dtheta + 2.f * sinPsi * Kperp2 * DsinPsi_dtheta) + 2.f * (sq(cosPsi) * Kpar + Kperp2 * sq(sinPsi)) * sinZeta * DsinZeta_dtheta;

        KK.DKrt_dr = (-Kperp1 + sq(cosPsi) * Kpar + Kperp2 * sq(sinPsi)) * (sinZeta * DcosZeta_dr + cosZeta * DsinZeta_dr) + cosZeta * sinZeta * (2.f * cosPsi * Kpar * DcosPsi_dr + sq(cosPsi) * dK_dr.x - dK_dr.y + sinPsi * (sinPsi * dK_dr.z + 2.f * Kperp2 * DsinPsi_dr));
        KK.DKtr_dt = (-Kperp1 + sq(cosPsi) * Kpar + Kperp2 * sq(sinPsi)) * (sinZeta * DcosZeta_dtheta + cosZeta * DsinZeta_dtheta) + cosZeta * sinZeta * (2.f * cosPsi * Kpar * DcosPsi_dtheta + 2.f * sinPsi * Kperp2 * DsinPsi_dtheta);
        KK.DKrp_dr = cosZeta * (Kperp2 - Kpar) * sinPsi * DcosPsi_dr + cosPsi * (Kperp2 - Kpar) * sinPsi * DcosZeta_dr + cosPsi * cosZeta * sinPsi * (dK_dr.z - dK_dr.x) + cosPsi * cosZeta * (Kperp2 - Kpar) * DsinPsi_dr;

        KK.DKtp_dt = (Kperp2 - Kpar) * (sinPsi * sinZeta * DcosPsi_dtheta + cosPsi * (sinZeta * DsinPsi_dtheta + sinPsi * DsinZeta_dtheta));
    } else {
        KK.rr = Diffusion_Coeff_heliosheat(index, qp, beta_R(qp.R, pt), KK.DKrr_dr);
    }
    // @formatter:on
    return KK;
}


/**
 * @brief Solving the square root of diffusion tensor in heliocentric spherical coordinates
 *
 * @param index
 * @param qp the quasi-particle
 * @param K Diffusion tensor
 * @param res 0 if ok; 1 if error
 * @return Square root of the diffusion tensor
 * @note This function is not implemented for the outer heliosphere: HZone >= Heliosphere.Nregions
 */
__device__ Tensor3D_t SquareRoot_DiffusionTerm(const Index_t &index, const QuasiParticle_t &qp, DiffusionTensor_t K,
                                               int *res) {
    Tensor3D_t D;

    K.rr = 2.f * K.rr;
    D.rr = sqrtf(K.rr); // g = sqrt(a)
    if (index.radial < Constants.Nregions) {
        K.tr = 2.f * K.tr / qp.r;
        K.tt = 2.f * K.tt / (qp.r * qp.r);
        K.pr = 2.f * K.pr / (qp.r * sinf(qp.th));
        K.pt = 2.f * K.pt / (qp.r * qp.r * sinf(qp.th));
        K.pp = 2.f * K.pp / (qp.r * qp.r * sinf(qp.th) * sinf(qp.th));

        D.tr = K.tr / D.rr;
        D.pr = K.pr / D.rr;
        D.tt = sqrtf(K.tt - D.tr * D.tr);
        D.pt = 1.f / D.tt * (K.pt - D.tr * D.pr);
        D.pp = sqrtf(K.pp - D.pr * D.pr - D.pt * D.pt);
    }

    if (const float sum = D.rr + D.tr + D.pr + D.tt + D.pt + D.pp; isnan(sum) || isinf(sum)) {
        // TODO -- check an other solution... see Pei et al 2010 or Kopp et al 2012
        // not implemented since such cases are rare
        *res = 1;
    } else {
        *res = 0;
    }

    return D;
}


/**
 * @brief Advective term of the SDE in heliocentric spherical coordinates
 *
 * @param index
 * @param qp the quasi-particle
 * @param K Diffusion tensor
 * @param pt Particle rest mass, atomic number, and mass number
 * @return Advective term
 */
__device__ vect3D_t AdvectiveTerm(const Index_t &index, const QuasiParticle_t &qp, const DiffusionTensor_t &K,
                                  const PartDescription_t pt) {
    vect3D_t AdvTerm = {2.f * K.rr / qp.r + K.DKrr_dr, 0, 0};

    if (index.radial < Constants.Nregions) {
        AdvTerm.r += K.tr / (qp.r * tanf(qp.th)) + K.DKtr_dt / qp.r;
        AdvTerm.th += K.tr / sq(qp.r) + K.tt / (tanf(qp.th) * sq(qp.r)) + K.DKrt_dr / qp.r + K.DKtt_dt / sq(qp.r);

        AdvTerm.phi += K.pr / (sq(qp.r) * sinf(qp.th)) + K.DKrp_dr / (qp.r * sinf(qp.th)) + K.DKtp_dt / (
            sq(qp.r) * sinf(qp.th));
        const auto [drift_r, drift_th, drift_phi] = Drift_PM89(index, qp, pt);
        AdvTerm.r -= drift_r;
        AdvTerm.th -= drift_th / qp.r;
        AdvTerm.phi -= drift_phi / (qp.r * sinf(qp.th));
    }

    AdvTerm.r -= SolarWindSpeed(index, qp);
    return AdvTerm;
}

/**
 * @brief Calculation of the energy loss term in heliocentric spherical coordinates
 *
 * @param index
 * @param qp the quasi-particle
 * @return Energy loss term
 */
__device__ float EnergyLoss(const Index_t &index, const QuasiParticle_t &qp) {
    if (index.radial < Constants.Nregions) {
        return 2.f / 3.f * SolarWindSpeed(index, qp) / qp.r * qp.R;
    }
    return 0;
}
