#include "GenComputation.cuh"
#include "HeliosphereModel.cuh"
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"
#include <cstdio>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions
#include <tuple>


/**
 * @brief Rescale the heliosphere boundaries to the effective heliosphere.
 *
 * @param Rbound Heliosphere boundaries to be rescaled
 * @param parts Initial position to be rescaled
 * @param index Index of the particle
 */
void RescaleToEffectiveHeliosphere(HeliosphereBoundRadius_t &Rbound, InitialPositions_t &parts, const unsigned index) {
    const float Rts_nose_realworld = Rbound.Rts_nose;
    const float Rhp_nose_realworld = Rbound.Rhp_nose;
    const float Rts_tail_realworld = Rbound.Rts_tail;
    const float Rhp_tail_realworld = Rbound.Rhp_tail;

    Rbound.Rts_nose = 100.;
    Rbound.Rts_tail = Rts_tail_realworld * Rbound.Rts_nose / Rts_nose_realworld;

    Rbound.Rhp_nose = Rbound.Rts_nose + (Rhp_nose_realworld - Rts_nose_realworld); //122.;
    Rbound.Rhp_tail = Rbound.Rts_tail + (Rhp_tail_realworld - Rts_tail_realworld); //Rhp_tail*Rhp/Rhp_realworld;

    const float HM_Rts_d = Boundary(parts.th[index], parts.phi[index], Rbound.Rts_nose, Rbound.Rts_tail);
    const float RW_Rts_d = Boundary(parts.th[index], parts.phi[index], Rts_nose_realworld, Rts_tail_realworld);
    if (const float Rdi_real = parts.r[index]; Rdi_real <= RW_Rts_d) parts.r[index] = Rdi_real / RW_Rts_d * HM_Rts_d;
    else parts.r[index] = HM_Rts_d + (Rdi_real - RW_Rts_d);
}


/**
 * @brief The K0 parameter is evaluated using the smoothed sunspot number as a proxy.
 *
 * @param p Solar polarity of HMF
 * @param SolarPhase Indicates the phase of the solar activity cycle (0=rising / 1=Declining)
 * @param ssn Smoothed sunspot number
 * @param GaussVar Gaussian variation (output)
 * @return K0 parameter
 */
float K0Fit_ssn(const int p, const int SolarPhase, const float ssn, float *GaussVar) {
    float k0;
    if (p > 0.) {
        if (SolarPhase == 0) {
            k0 = 0.0002743f - 2.11e-6f * ssn + 1.486e-8f * sq(ssn) - 3.863e-11f * sq(ssn) * ssn;
            *GaussVar = 0.1122;
        } else {
            k0 = 0.0002787f - 1.66e-6f * ssn + 4.658e-9f * sq(ssn) - 6.673e-12f * sq(ssn) * ssn;
            *GaussVar = 0.1324f;
        }
    } else {
        if (SolarPhase == 0) {
            k0 = 0.0003059f - 2.51e-6f * ssn + 1.284e-8f * sq(ssn) - 2.838e-11f * sq(ssn) * ssn;
            *GaussVar = 0.1097;
        } else {
            k0 = 0.0002876f - 3.715e-6f * ssn + 2.534e-8f * sq(ssn) - 5.689e-11f * sq(ssn) * ssn;
            *GaussVar = 0.14;
        }
    }
    return k0;
}

/**
 * @brief The K0 parameter is evaluated using the McMurdo neutron monitor counting rate as a proxy.
 *
 * @param NMC Neutron monitor counting rate from McMurdo
 * @param GaussVar Gaussian variation (output)
 * @return K0 parameter
 */
float K0Fit_NMC(const float NMC, float *GaussVar) {
    *GaussVar = 0.1045;
    return expf(-10.83f - 0.0041f * NMC + 4.52e-5f * sq(NMC));
}

/**
 * @brief Correction factor to K0 for the Kparallel.
 *
 * This correction is introduced to account for the fact that K0 is evaluated
 * with a model not including particle drift.
 * Thus, the value need a correction once to be used in present model.
 *
 * @param p Solar polarity of HMF
 * @param q Signum of particle charge
 * @param SolarPhase Indicates the phase of the solar activity cycle (0=rising / 1=Declining)
 * @param tilt Tilt angle of neutral sheet (in degree)
 * @return float The correction factor
 */
float K0CorrFactor(const int p, const int q, const int SolarPhase, const float tilt) {
    // TODO: Move constants to a configuration file
#ifndef K0Corr_maxv
#define K0Corr_maxv 1.5f
#endif
#ifndef K0Corr_minv
#define K0Corr_minv 1.f
#endif
#ifndef K0Corr_p0_asc
#define K0Corr_p0_asc 18.f
#endif
#ifndef K0Corr_p1_asc
#define K0Corr_p1_asc 40.f
#endif
#ifndef K0Corr_p0_des
#define K0Corr_p0_des 5.f
#endif
#ifndef K0Corr_p1_des
#define K0Corr_p1_des 53.f
#endif
#ifndef K0Corr_maxv_neg
#define K0Corr_maxv_neg 0.7f
#endif
#ifndef K0Corr_p0_asc_neg
#define K0Corr_p0_asc_neg 5.8f
#endif
#ifndef K0Corr_p1_asc_neg
#define K0Corr_p1_asc_neg 47.f
#endif
#ifndef K0Corr_p0_des_neg
#define K0Corr_p0_des_neg 5.8f
#endif
#ifndef K0Corr_p1_des_neg
#define K0Corr_p1_des_neg 58.f
#endif

    if (q > 0) {
        if (q * p > 0) {
            if (SolarPhase == 0) {
                // Ascending
                return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_asc, K0Corr_p0_asc, tilt);
            }
            // Descending
            return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_des, K0Corr_p0_des, tilt);
        }
        return 1;
    }
    if (q < 0) {
        if (q * p > 0) {
            if (SolarPhase == 0) {
                // Ascending
                return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_asc, K0Corr_p0_asc, tilt);
            }
            // Descending
            return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_des, K0Corr_p0_des, tilt);
        }
        if (SolarPhase == 0) {
            // Ascending
            return SmoothTransition(K0Corr_maxv_neg, K0Corr_minv, K0Corr_p1_asc_neg, K0Corr_p0_asc_neg, tilt);
        }
        // Descending
        return SmoothTransition(K0Corr_maxv_neg, K0Corr_minv, K0Corr_p1_des_neg, K0Corr_p0_des_neg, tilt);
    }
    return 1;
}


/**
 * @brief Evaluate diffusion parameter from fitting procedures.
 *
 * @param IsHighActivityPeriod Flag to indicate if the period is of high solar activity
 * @param p Solar polarity of HMF
 * @param q Signum of particle charge
 * @param SolarPhase Indicates the phase of the solar activity cycle (0=rising / 1=Declining)
 * @param tilt Tilt angle of neutral sheet (in degree)
 * @param NMC Neutron monitor counting rate from McMurdo
 * @param ssn Smoothed sunspot number
 * @param verbose Verbosity level
 * @return output x = k0_paral (parallel k0), y = k0_perp (perpendicular k0), z = GaussVar (Gaussian variation)
 * k0_paral is corrected by a correction factor
 */
std::tuple<float, float, float> EvalK0(const bool IsHighActivityPeriod, const int p, const int q, const int SolarPhase,
                                       const float tilt,
                                       const float NMC, const float ssn, const unsigned char verbose = 0) {
    float K_par = K0CorrFactor(p, q, SolarPhase, tilt), K_perp, Gauss;

    if (IsHighActivityPeriod && NMC > 0) {
        K_perp = K0Fit_NMC(NMC, &Gauss);
        K_par *= K_perp;
    } else {
        if (verbose >= VERBOSE_med && IsHighActivityPeriod && NMC == 0) {
            fprintf(
                stderr,
                "WARNING:: High Activity period require NMC variable setted with value >0, used ssn instead.\n");
        }
        K_perp = K0Fit_ssn(p, SolarPhase, ssn, &Gauss);
        K_par *= K_perp;
    }
    return {K_par, K_perp, Gauss};
}

/**
 * @brief Evaluation of g_low parameter (for K0 parallel).
 *
 * @param SolarPhase Indicates the phase of the solar activity cycle (0=rising / 1=Declining)
 * @param Polarity Solar polarity of HMF
 * @param tilt Tilt angle of neutral sheet (in degree)
 * @return g_low parameter
 */
float g_low(const int SolarPhase, const int Polarity, const float tilt) {
    // TODO: Move constants to a configuration file
#ifndef MaxValueOf_g_low_pos
#define MaxValueOf_g_low_pos 0.6f
#endif
#ifndef CAB_TransPoint_des_pos
#define CAB_TransPoint_des_pos 45
#endif
#ifndef CAB_smoothness_des_pos
#define CAB_smoothness_des_pos 5.f
#endif
#ifndef CAB_TransPoint_asc_pos
#define CAB_TransPoint_asc_pos 60
#endif
#ifndef CAB_smoothness_asc_pos
#define CAB_smoothness_asc_pos 9.f
#endif
#ifndef MaxValueOf_g_low_neg
#define MaxValueOf_g_low_neg 0.5f
#endif
#ifndef CAB_TransPoint_des_neg
#define CAB_TransPoint_des_neg 45
#endif
#ifndef CAB_smoothness_des_neg
#define CAB_smoothness_des_neg 10.f
#endif
#ifndef CAB_TransPoint_asc_neg
#define CAB_TransPoint_asc_neg 60.f
#endif
#ifndef CAB_smoothness_asc_neg
#define CAB_smoothness_asc_neg 9.f
#endif
    float g_low = 0;
    float MaxValueOf_g_low, CAB_TransPoint_des, CAB_smoothness_des, CAB_TransPoint_asc, CAB_smoothness_asc;
    if (Polarity > 0) {
        MaxValueOf_g_low = MaxValueOf_g_low_pos;
        CAB_TransPoint_des = CAB_TransPoint_des_pos;
        CAB_smoothness_des = CAB_smoothness_des_pos;
        CAB_TransPoint_asc = CAB_TransPoint_asc_pos;
        CAB_smoothness_asc = CAB_smoothness_asc_pos;
    } else {
        MaxValueOf_g_low = MaxValueOf_g_low_neg;
        CAB_TransPoint_des = CAB_TransPoint_des_neg;
        CAB_smoothness_des = CAB_smoothness_des_neg;
        CAB_TransPoint_asc = CAB_TransPoint_asc_neg;
        CAB_smoothness_asc = CAB_smoothness_asc_neg;
    }

    if (SolarPhase == 1) {
        g_low = MaxValueOf_g_low * SmoothTransition(1, 0, CAB_TransPoint_des, CAB_smoothness_des, tilt);
    } else {
        g_low = MaxValueOf_g_low * SmoothTransition(1, 0, CAB_TransPoint_asc, CAB_smoothness_asc, tilt);
    }
    return g_low;
}

/**
 * @brief Evalutation of rconst parameter (for K0 parallel).
 *
 * @param SolarPhase Indicates the phase of the solar activity cycle (0=rising / 1=Declining)
 * @param Polarity Solar polarity of HMF
 * @param tilt Tilt angle of neutral sheet (in degree)
 * @return rconst parameter
 */
float rconst(const int SolarPhase, const int Polarity, const float tilt) {
    // TODO: Spostare le costanti in un file di configurazione
#ifndef MaxValueOf_rconst
#define MaxValueOf_rconst 4
#endif
#ifndef rconst_TransPoint_des_pos
#define rconst_TransPoint_des_pos 45
#endif
#ifndef rconst_smoothness_des_pos
#define rconst_smoothness_des_pos 5.f
#endif
#ifndef rconst_TransPoint_asc_pos
#define rconst_TransPoint_asc_pos 60
#endif
#ifndef rconst_smoothnesst_asc_pos
#define rconst_smoothness_asc_pos 9.f
#endif
#ifndef rconst_TransPoint_des_neg
#define rconst_TransPoint_des_neg 45
#endif
#ifndef rconst_smoothness_des_neg
#define rconst_smoothness_des_neg 10.f
#endif
#ifndef rconst_TransPoint_asc_neg
#define rconst_TransPoint_asc_neg 60
#endif
#ifndef rconst_smoothness_asc_neg
#define rconst_smoothness_asc_neg 9.f
#endif
    float rconst = 0;
    float rconst_TransPoint_des, rconst_smoothness_des, rconst_TransPoint_asc, rconst_smoothness_asc;
    if (Polarity > 0) {
        rconst_TransPoint_des = rconst_TransPoint_des_pos;
        rconst_smoothness_des = rconst_smoothness_des_pos;
        rconst_TransPoint_asc = rconst_TransPoint_asc_pos;
        rconst_smoothness_asc = rconst_smoothness_asc_pos;
    } else {
        rconst_TransPoint_des = rconst_TransPoint_des_neg;
        rconst_smoothness_des = rconst_smoothness_des_neg;
        rconst_TransPoint_asc = rconst_TransPoint_asc_neg;
        rconst_smoothness_asc = rconst_smoothness_asc_neg;
    }

    if (SolarPhase == 1) {
        rconst = SmoothTransition(MaxValueOf_rconst, 1, rconst_TransPoint_des, rconst_smoothness_des, tilt);
    } else {
        rconst = SmoothTransition(MaxValueOf_rconst, 1, rconst_TransPoint_asc, rconst_smoothness_asc, tilt);
    }
    return rconst;
}

/**
 * @brief Evaluation of the diffusion tensor in the HMF frame, i.e. k0 parallel and k0 perpendicular.
 *
 * @param index Initial zone in the heliosphere
 * @param qp
 * @param beta v/c
 * @param GaussRndNumber Random number with normal distribution
 * @param dK_dr Output parameter for the derivative of K with respect to r
 * @param params
 * @return x Kparallel
 * @return y Kperp_1
 * @return z Kperp_2
 */
__device__ float3 Diffusion_Tensor_In_HMF_Frame(const Index_t &index, const QuasiParticle_t &qp, const float beta,
                                                const float GaussRndNumber, float3 &dK_dr,
                                                const SimulationParametrizations_t params) {
    float3 Ktensor;

    const int high_activity = Constants.IsHighActivityPeriod[index.period] ? 0 : 1;
    const float k0_paral = params.params[index.param].heliosphere[index.combined()].k0_paral[high_activity];
    const float k0_perp = params.params[index.param].heliosphere[index.combined()].k0_perp[high_activity];
    const float GaussVar = params.params[index.param].heliosphere[index.combined()].GaussVar[high_activity];
    const float g_low = Constants.heliosphere_properties[index.combined()].g_low;
    const float rconst = Constants.heliosphere_properties[index.combined()].rconst;


    // Kpar = k0 * beta/3 * (P/1GV + glow)*( Rconst+r/1AU) with k0 gaussian distributed
    dK_dr.x = (k0_paral + GaussRndNumber * GaussVar * k0_paral) * beta / 3.f * (qp.R + g_low);
    Ktensor.x = dK_dr.x * (rconst + qp.r);

    // TODO: Move constants to a configuration file
#ifndef rho_1
#define rho_1 0.065f // Kpar/Kperp (ex Kp0)
#endif
#ifndef PolarEnhanc
#define PolarEnhanc 2 // polar enhancement in polar region
#endif

    // Kperp1 = rho_1(theta)* k0 * beta/3 * (P/1GV + glow)*( Rconst+r/1AU)
    dK_dr.y = rho_1 * k0_perp * beta / 3.f * (qp.R + g_low) * (fabsf(cosf(qp.th)) > CosPolarZone ? PolarEnhanc : 1.f);
    Ktensor.y = dK_dr.y * (rconst + qp.r);

    // Kperp2 = rho_2 * k0 * beta/3 * (P/1GV + glow)*( Rconst+r/1AU) with rho_2=rho_1
    dK_dr.z = rho_1 * k0_perp * beta / 3.f * (qp.R + g_low);
    Ktensor.z = dK_dr.z * (rconst + qp.r);

    return Ktensor;
}

/**
 * @brief Evaluation of the diffusion tensor in the HMF frame, i.e. K0 parallel and K0 perpendicular.
 *
 * @param index Initial zone in the heliosphere
 * @param qp QuasiParticle
 * @param beta v/c
 * @param dK_dr Output parameter for the derivative of K with respect to r
 * @return Diffusion coefficient
 */
__device__ float Diffusion_Coeff_heliosheat(const Index_t &index, const QuasiParticle_t &qp, const float beta,
                                            float &dK_dr) {
    dK_dr = 0.;
    // if around 5 AU from Heliopause, apply diffusion barrier
    const float RhpDirection = Boundary(qp.th, qp.phi, Constants.RadBoundary_effe[index.period].Rhp_nose,
                                        Constants.RadBoundary_effe[index.period].Rhp_tail);
    // TODO: Spostare le costanti in un file di configurazione
#ifndef HPB_SupK
#define HPB_SupK 50 // suppressive factor at barrier
#endif
#ifndef HP_width
#define HP_width 2 // amplitude in AU of suppressive factor at barrier
#endif
#ifndef HP_SupSmooth
#define HP_SupSmooth 3e-2 // smoothness of suppressive factor at barrier
#endif

    if (qp.r > RhpDirection - 5) {
        return Constants.heliosheat_properties[index.period].k0 * beta * qp.R * SmoothTransition(
                   1, 1.f / HPB_SupK, RhpDirection - HP_width / 2.f,
                   HP_SupSmooth, qp.r);
    }
    return Constants.heliosheat_properties[index.period].k0 * beta * qp.R;
}
