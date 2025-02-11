#include "GenComputation.cuh"
#include "HeliosphereModel.cuh"
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"
#include <cstdio>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions


/////////////////////////////////////////////////////////////////////////////
////////////////// Diffusion Parameters Description /////////////////////////
/////////////////////////////////////////////////////////////////////////////
void RescaleToEffectiveHeliosphere(HeliosphereBoundRadius_t &Rbound, vect3D_t &part) {
    /* * description: create an effective heliosphere of 100 AU. thisis due to the fact that K0 parameters are tuned on such dimension.
         \param  Rbound heliospher boundaries to be rescaled
         \param  part   initial position to be rescaled
     */
    const float Rts_nose_realworld = Rbound.Rts_nose;
    const float Rhp_nose_realworld = Rbound.Rhp_nose;
    const float Rts_tail_realworld = Rbound.Rts_tail;
    const float Rhp_tail_realworld = Rbound.Rhp_tail;

    Rbound.Rts_nose = 100.;
    Rbound.Rts_tail = Rts_tail_realworld * Rbound.Rts_nose / Rts_nose_realworld;

    Rbound.Rhp_nose = Rbound.Rts_nose + (Rhp_nose_realworld - Rts_nose_realworld); //122.;
    Rbound.Rhp_tail = Rbound.Rts_tail + (Rhp_tail_realworld - Rts_tail_realworld); //Rhp_tail*Rhp/Rhp_realworld;

    const float HM_Rts_d = Boundary(part.th, part.phi, Rbound.Rts_nose, Rbound.Rts_tail);
    const float RW_Rts_d = Boundary(part.th, part.phi, Rts_nose_realworld, Rts_tail_realworld);
    if (const float Rdi_real = part.r; Rdi_real <= RW_Rts_d) part.r = Rdi_real / RW_Rts_d * HM_Rts_d;
    else part.r = HM_Rts_d + (Rdi_real - RW_Rts_d);
}


float K0Fit_ssn(const int p, const int SolarPhase, const float ssn, float *GaussVar) {
    /*Authors: 2011 Stefano - update 2012 Stefano - update 2015 GLV*/
    /* * description: K0 evaluated using ssn as a proxy
         \param p            solar polarity of HMF
         \param SolarPhase   0=rising / 1=Declining phase of solar activity cycle
         \param ssn          smoothed sunspot number
         \param *GaussVar    gaussian variation (output)
     */
    float k0;
    if (p > 0.) {
        if (SolarPhase == 0)/*Rising*/ {
            k0 = 0.0002743f - 2.11e-6f * ssn + 1.486e-8f * sq(ssn) - 3.863e-11f * sq(ssn) * ssn;
            *GaussVar = 0.1122;
        } else /*Declining*/ {
            k0 = 0.0002787f - 1.66e-6f * ssn + 4.658e-9f * sq(ssn) - 6.673e-12f * sq(ssn) * ssn;
            *GaussVar = 0.1324f;
        }
    } else {
        if (SolarPhase == 0)/*Rising*/ {
            k0 = 0.0003059f - 2.51e-6f * ssn + 1.284e-8f * sq(ssn) - 2.838e-11f * sq(ssn) * ssn;
            *GaussVar = 0.1097;
        } else /*Declining*/ {
            k0 = 0.0002876f - 3.715e-6f * ssn + 2.534e-8f * sq(ssn) - 5.689e-11f * sq(ssn) * ssn;
            *GaussVar = 0.14;
        }
    }
    return k0;
}

float K0Fit_NMC(const float NMC, float *GaussVar) {
    /*Authors: 2015 GLV*/
    /* * description: K0 evaluated using Mc Murdo NM counts as a proxy
                      only for High Activity, defined as Tilt L >48deg
       \param NMC          Neutron monitor counting rate from Mc Murdo
       \param *GaussVar    gaussian variation (output)
    */
    *GaussVar = 0.1045;
    return expf(-10.83f - 0.0041f * NMC + 4.52e-5f * sq(NMC));
}

float K0CorrFactor(const int p, const int q, const int SolarPhase, const float tilt) {
    /*Authors: 2017 Stefano */
    /* * description: Correction factor to K0 for the Kparallel. This correction is introduced
                      to account for the fact that K0 is evaluated with a model not including particle drift.
                      Thus, the value need a correction once to be used in present model
        \param p            solar polarity of HMF
        \param q            signum of particle charge
        \param SolarPhase   0=rising / 1=Declining phase of solar activity cycle
        \param tilt         Tilt angle of neutral sheet (in degree)
    */
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
                //ascending
                return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_asc, K0Corr_p0_asc, tilt);
            }
            //descending
            return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_des, K0Corr_p0_des, tilt);
        }
        return 1;
    }
    if (q < 0) {
        if (q * p > 0) {
            if (SolarPhase == 0) {
                //ascending
                return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_asc, K0Corr_p0_asc, tilt);
            }
            //descending
            return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_des, K0Corr_p0_des, tilt);
        }
        if (SolarPhase == 0) {
            //ascending
            return SmoothTransition(K0Corr_maxv_neg, K0Corr_minv, K0Corr_p1_asc_neg, K0Corr_p0_asc_neg, tilt);
        }
        //descending
        return SmoothTransition(K0Corr_maxv_neg, K0Corr_minv, K0Corr_p1_des_neg, K0Corr_p0_des_neg, tilt);
    }
    return 1;
}


float3 EvalK0(const bool IsHighActivityPeriod, const int p, const int q, const int SolarPhase, const float tilt,
              const float NMC, const float ssn,
              const unsigned char verbose = 0) {
    /*Authors: 2022 Stefano */
    /* * description: Evaluate diffusion parameter from fitting procedures.
        \param p            solar polarity of HMF
        \param q            signum of particle charge
        \param SolarPhase   0=rising / 1=Declining phase of solar activity cycle
        \param tilt         Tilt angle of neutral sheet (in degree)
        \return x = k0_paral
                y = k0_perp
                z = GaussVar
    */
    float3 output;
    output.x = K0CorrFactor(p, q, SolarPhase, tilt); // k0_paral is corrected by a correction factor
    ////////////////////////////////////////////////
    // printf("-- p: %d q: %d phase: %d tilt: %e ssn: %e NMC: %e \n",p,q,SolarPhase,tilt,ssn,NMC);
    // printf("-- K0CorrF: %e \n",output.x);
    // printf("-- IsHighActivityPeriod %d \n",IsHighActivityPeriod);
    ////////////////////////////////////////////////
    if (IsHighActivityPeriod && NMC > 0) {
        output.y = K0Fit_NMC(NMC, &output.z);
        output.x *= output.y;
    } else {
        if (verbose >= VERBOSE_med && IsHighActivityPeriod && NMC == 0) {
            fprintf(
                stderr,
                "WARNING:: High Activity period require NMC variable setted with value >0, used ssn instead.\n");
        }
        output.y = K0Fit_ssn(p, SolarPhase, ssn, &output.z);
        output.x *= output.y;
    }
    ////////////////////////////////////////////////
    // printf("-- K0 paral: %e \n",output.x);
    // printf("-- K0 perp : %e \n",output.y);
    ////////////////////////////////////////////////
    return output;
}

float g_low(const int SolarPhase, const int Polarity, const float tilt) {
    /*Authors: 2022 Stefano - update 2023 GLV*/
    /* * description: evaluate g_low parameter (for Kparallel).
        \param SolarPhase   0=rising / 1=Declining phase of solar activity cycle
        \param tilt         Tilt angle of neutral sheet (in degree)
        \return g_low
    */
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

float rconst(const int SolarPhase, const int Polarity, const float tilt) {
    /*Authors: 2022 Stefano - update 2023 GLV*/
    /* * description: evaluate rconst parameter (for Kparallel).
        \param SolarPhase   0=rising / 1=Declining phase of solar activity cycle
        \param tilt         Tilt angle of neutral sheet (in degree)
        \return rconst
    */
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

__device__ float3 Diffusion_Tensor_In_HMF_Frame(const unsigned int InitZone, const signed int HZone, const float r,
                                                const float theta, const float beta, const float P,
                                                const float GaussRndNumber, float3 &dK_dr) {
    /*Authors: 2022 Stefano */
    /* * description: evaluate the diffusion tensor in the HMF frame, i.e. Kparallel & Kperpendicular.
        \param HZone   Zone in the Heliosphere
        \param r      solar distance
        \param theta  solar colatitude
        \param beta   v/c
        \param P      Particle rigidity
        \param GaussRndNumber Random number with normal distribution
        \return x Kparallel
                y Kperp_1
                z Kperp_2
    */
    float3 Ktensor;
    // HeliosphereZoneProperties_t ThisZone=LIM[HZone+InitZone];

    const int high_activity = Heliosphere.IsHighActivityPeriod[InitZone] ? 0 : 1;
    const float k0_paral = LIM[HZone + InitZone].k0_paral[high_activity];
    const float k0_perp = LIM[HZone + InitZone].k0_perp[high_activity];
    const float GaussVar = LIM[HZone + InitZone].GaussVar[high_activity];
    const float g_low = LIM[HZone + InitZone].g_low;
    const float rconst = LIM[HZone + InitZone].rconst;


    // Kpar = k0 * beta/3 * (P/1GV + glow)*( Rconst+r/1AU) with k0 gaussian distributed
    dK_dr.x = (k0_paral + GaussRndNumber * GaussVar * k0_paral) * beta / 3.f * (P + g_low);
    Ktensor.x = dK_dr.x * (rconst + r);

#ifndef rho_1
#define rho_1 0.065f // Kpar/Kperp (ex Kp0)
#endif
#ifndef PolarEnhanc
#define PolarEnhanc 2 // polar enhancement in polar region
#endif

    // Kperp1 = rho_1(theta)* k0 * beta/3 * (P/1GV + glow)*( Rconst+r/1AU)
    dK_dr.y = rho_1 * k0_perp * beta / 3.f * (P + g_low) * (fabsf(cosf(theta)) > CosPolarZone ? PolarEnhanc : 1.f);
    Ktensor.y = dK_dr.y * (rconst + r);

    // Kperp2 = rho_2 * k0 * beta/3 * (P/1GV + glow)*( Rconst+r/1AU) with rho_2=rho_1
    dK_dr.z = rho_1 * k0_perp * beta / 3.f * (P + g_low);
    Ktensor.z = dK_dr.z * (rconst + r);

    return Ktensor;
}

__device__ float Diffusion_Coeff_heliosheat(const unsigned int HZone, const float r, const float th, const float phi,
                                            const float beta, const float P, float &dK_dr) {
    /*Authors: 2022 Stefano */
    /* * description: evaluate the diffusion tensor in the HMF frame, i.e. Kparallel & Kperpendicular.
        \param HZone   Zone in the Heliosphere
        \param r      solar distance
        \param beta   v/c
        \param P      Particle rigidity
        \return x diffusion coeff
    */
    dK_dr = 0.;
    // if around 5 AU from Heliopause, apply diffusion barrier
    const float RhpDirection = Boundary(th, phi, Heliosphere.RadBoundary_effe[HZone].Rhp_nose,
                                        Heliosphere.RadBoundary_effe[HZone].Rhp_tail);
#ifndef HPB_SupK
#define HPB_SupK 50 // suppressive factor at barrier
#endif
#ifndef HP_width
#define HP_width 2 // amplitude in AU of suppressive factor at barrier
#endif
#ifndef HP_SupSmooth
#define HP_SupSmooth 3e-2 // smoothness of suppressive factor at barrier
#endif
    if (r > RhpDirection - 5) {
        return HS[HZone].k0 * beta * P * SmoothTransition(1, 1. / HPB_SupK, RhpDirection - HP_width / 2.f,
                                                          HP_SupSmooth, r);
    }
    return HS[HZone].k0 * beta * P;
}
