#include "GenComputation.cuh"
#include "HelModVariableStructure.cuh"
#include "VariableStructure.cuh"
#include "MagneticDrift.cuh"
#include "SolarWind.cuh"

/**
 * @brief Compute Gamma_Bfield
 * @param r
 * @param th
 * @param Vsw
 * @return float
*/
__device__ float Gamma_Bfield(const float r, const float th, const float Vsw) {
    return Omega * (r - rhelio) * sinf(th) / Vsw;
}

/**
 * @brief Compute beta_R field
 * @param r
 * @param th
 * @return float
*/
__device__ float delta_Bfield(const float r, const float th) {
    return r / rhelio * delta_m / sinf(th);;
}

/**
 * @brief Evaluate drift velocity vector, including Neutral sheetdrift. This drift is based on Potgieter e Mooral Model (1989), this model was modified to include a theta-component in Bfield.
 * @param index
 * @param qp the quasi-particle
 * @param pt the particle description
 * @return vect3D_t drift velocity vector
 * @note global CR drifts are believed to be reduced due to the presence of turbulence (scattering; e.g., Minnie et al. 2007). This is incorporated into the modulation
        model by defining the drift reduction factor. Strauss et al 2011, Minnie et al 2007 Burger et al 2000
*/
__device__ vect3D_t Drift_PM89(const Index_t &index, const QuasiParticle_t &qp, const PartDescription_t pt) {
//@formatter:off
    vect3D_t v;
    const float IsPolarRegion = fabsf(cosf(qp.th)) > CosPolarZone;
    const float Ka = safeSign(pt.Z) * GeV * beta_R(qp.R, pt) * qp.R / 3;
    const float Asun = Constants.heliosphere_properties[index.combined()].Asun;
    const float dV_dth = DerivativeOfSolarWindSpeed_dtheta(index, qp);
    const float TiltAngle = Constants.heliosphere_properties[index.combined()].TiltAngle;
    // .. Scaling factor of drift in 2D approximation.  to account Neutral sheet
    float fth = 0; /* scaling function of drift vel */
    float Dftheta_dtheta = 0;
    const float TiltPos_r = qp.r;
    const float TiltPos_th = Pi / 2.f - TiltAngle;
    const float TiltPos_phi = qp.phi;
    float Vsw = SolarWindSpeed(index, {TiltPos_r, TiltPos_th, TiltPos_phi});
    const float dthetans = fabsf(GeV / (SoL * aum) * (2.f * qp.r * qp.R) / (Asun * sqrtf(1 + sq(Gamma_Bfield(qp.r, TiltPos_th, Vsw)) + IsPolarRegion * sq(delta_Bfield(qp.r, TiltPos_th)))));

    if (const float theta_mez = Pi / 2.f - 0.5f * sinf(fminf(TiltAngle + dthetans, Pi / 2.f));
        theta_mez < Pi / .2f) {
        const float a_f = acosf(Pi / (2.f * theta_mez) - 1);
        fth = 1.f / a_f * atanf((1.f - 2.f * qp.th / Pi) * tanf(a_f));
        Dftheta_dtheta = -2.f * tanf(a_f) / (a_f * Pi * (1.f + (1 - 2.f * qp.th / Pi) * (1 - 2.f * qp.th / Pi) * tanf(a_f) * tanf(a_f)));
    } else {
        fth = sign(qp.th - Pi / 2.f);
    }

    Vsw = SolarWindSpeed(index, qp);

    const float dm = IsPolarRegion ? delta_m : 0;

    const float E = dm * dm * qp.r * qp.r * Vsw * Vsw + Omega * Omega * rhelio * rhelio * (qp.r - rhelio) * (qp.r - rhelio) * sinf(qp.th) * sinf(qp.th) * sinf(qp.th) * sinf(qp.th) + rhelio * rhelio * Vsw * Vsw * sinf(qp.th) * sinf(qp.th);
    float C = fth * sinf(qp.th) * Ka * qp.r * rhelio / (Asun * E * E);
    C *= qp.R * qp.R / (qp.R * qp.R + Constants.heliosphere_properties[index.combined()].P0d * Constants.heliosphere_properties[index.combined()].P0d);

    v.r = -C * Omega * rhelio * 2 * (qp.r - rhelio) * sinf(qp.th) * ((2 * dm * dm * qp.r * qp.r + rhelio * rhelio * sinf(qp.th) * sinf(qp.th)) * Vsw * Vsw * Vsw * cosf(qp.th) - .5f * (dm * dm * qp.r * qp.r * Vsw * Vsw - Omega * Omega * rhelio * rhelio * (qp.r - rhelio) * (qp.r - rhelio) * sinf(qp.th) * sinf(qp.th) * sinf(qp.th) * sinf(qp.th) + rhelio * rhelio * Vsw * Vsw * sinf(qp.th) * sinf(qp.th)) * sinf(qp.th) * dV_dth);
    v.th = -C * Omega * rhelio * Vsw * sinf(qp.th) * sinf(qp.th) * (2 * qp.r * (qp.r - rhelio) * (dm * dm * qp.r * Vsw * Vsw + Omega * Omega * rhelio * rhelio * (qp.r - rhelio) * sinf(qp.th) * sinf(qp.th) * sinf(qp.th) * sinf(qp.th)) - (4 * qp.r - 3 * rhelio) * E);
    v.phi = 2 * C * Vsw * (-dm * dm * qp.r * qp.r * (dm * qp.r + rhelio * cosf(qp.th)) * Vsw * Vsw * Vsw + 2 * dm * qp.r * E * Vsw - Omega * Omega * rhelio * rhelio * (qp.r - rhelio) * sinf(qp.th) * sinf(qp.th) * sinf(qp.th) * sinf(qp.th) * (dm * qp.r * qp.r * Vsw - rhelio * (qp.r - rhelio) * Vsw * cosf(qp.th) + rhelio * (qp.r - rhelio) * sinf(qp.th) * dV_dth));

    C = Vsw * Dftheta_dtheta * sinf(qp.th) * sinf(qp.th) * Ka * qp.r * rhelio * rhelio / (Asun * E);
    C *= qp.R * qp.R / (qp.R * qp.R + Constants.heliosphere_properties[index.combined()].P0dNS * Constants.heliosphere_properties[index.combined()].P0dNS);

    v.r += -C * Omega * sinf(qp.th) * (qp.r - rhelio);

    v.phi += -C * Vsw;

    const float HighRigiSupp = Constants.heliosphere_properties[index.combined()].plateau + (1.f - Constants.heliosphere_properties[index.combined()].plateau) / (1.f + expf(HighRigiSupp_smoothness * (qp.R - HighRigiSupp_TransPoint)));
    v.r *= HighRigiSupp;
    v.th *= HighRigiSupp;
    v.phi *= HighRigiSupp;
    //@formatter:on
    return v;
}

/**
* @brief Evaluate drift suppression factor for the drift
* @param WhichDrift 0 - Regular drift, 1 - NS drift
* @param SolarPhase 0 - Ascending, 1 - Descending
* @param TiltAngleDeg Tilt angle in degrees
* @param ssn Solar modulation parameter
* @return float drift suppression factor
*/
float EvalP0DriftSuppressionFactor(const int WhichDrift, const int SolarPhase, const float TiltAngleDeg,
                                   const float ssn) {
    float InitialVal, FinalVal, CenterOfTransition, smoothness;

    if (WhichDrift == 0) {
        InitialVal = TDDS_P0d_Ini;
        FinalVal = TDDS_P0d_Fin;
        if (SolarPhase == 0) {
            CenterOfTransition = TDDS_P0d_CoT_asc;
            smoothness = TDDS_P0d_Smt_asc;
        } else {
            CenterOfTransition = TDDS_P0d_CoT_des;
            smoothness = TDDS_P0d_Smt_des;
        }
    } else {
        InitialVal = TDDS_P0dNS_Ini;
        FinalVal = ssn / SSNScalF;
        if (SolarPhase == 0) {
            CenterOfTransition = TDDS_P0dNS_CoT_asc;
            smoothness = TDDS_P0dNS_Smt_asc;
        } else {
            CenterOfTransition = TDDS_P0dNS_CoT_des;
            smoothness = TDDS_P0dNS_Smt_des;
        }
    }
    return SmoothTransition(InitialVal, FinalVal, CenterOfTransition, smoothness, TiltAngleDeg);
}

/**
* @brief Evaluate high rigidity drift suppression plateau
* @param SolarPhase 0 - Ascending, 1 - Descending
* @param TiltAngleDeg Tilt angle in degrees
* @return float drift suppression factor
*/
float EvalHighRigidityDriftSuppression_plateau(const int SolarPhase, const float TiltAngleDeg) {
    // Plateau time dependence
    float CenterOfTransition, smoothness;
    if (SolarPhase == 0) {
        CenterOfTransition = HRS_TransPoint_asc;
        smoothness = HRS_smoothness_asc;
    } else {
        CenterOfTransition = HRS_TransPoint_des;
        smoothness = HRS_smoothness_des;
    }
    return 1.f - SmoothTransition(1., 0., CenterOfTransition, smoothness, TiltAngleDeg);
}
