#include "GenComputation.cuh"
#include "HelModVariableStructure.cuh"
#include "VariableStructure.cuh"
#include "MagneticDrift.cuh"
#include "SolarWind.cuh"


////////////////////////////////////////////////////////////////
//..... Magnetic Drift Model ..................................
//.... Potgieter Mooral 1985 - See Burger&Hatttingh 1995
////////////////////////////////////////////////////////////////
__device__ float Gamma_Bfield(const float r, const float th, const float Vsw) {
    return Omega * (r - rhelio) * sinf(th) / Vsw;
}

__device__ float delta_Bfield(const float r, const float th) {
    return r / rhelio * delta_m / sinf(th);;
}

__device__ vect3D_t Drift_PM89(const Index_t &index, const QuasiParticle_t &qp, const PartDescription_t pt,
                               const HeliosphereZoneProperties_t *LIM) {
    /*Authors: 2022 Stefano */
    /* * description: Evalaute drift velocity vector, including Neutral sheetdrift. This drift is based on Potgieter e Mooral Model (1989), this model was modified to include a theta-component in Bfield.
     *  Imporant warning: the model born as 2D model for Pure Parker Field, originally this not include theta-component in Bfield.
     *  The implementation can be not fully rigorous,
        \param HZone   Zone in the Heliosphere
        \param part    particle position and energy
        \param pt      particle properties (Z,A,T0)
        \return drift velocity vector including Neutral sheet drift
  
        Additionally, global CR drifts are believed to be reduced due to the presence of turbulence (scattering; e.g., Minnie et al. 2007). This is incorporated into the modulation
        model by defining the drift reduction factor.
        See:  Strauss et al 2011, Minnie et al 2007 Burger et al 2000
     */
    vect3D_t v;
    const float IsPolarRegion = fabsf(cosf(qp.th)) > CosPolarZone;
    const float Ka = safeSign(pt.Z) * GeV * beta_R(qp.R, pt) * qp.R / 3;
    /* sign(Z)*beta*P/3 constant part of antisymmetric diffusion coefficient */
    // pt.A*sqrt(Ek*(Ek+2*pt.T0))/pt.Z
    const float Asun = LIM[index.combined()].Asun; /* Magnetic Field Amplitude constant / aum^2*/
    const float dV_dth = DerivativeOfSolarWindSpeed_dtheta(index, qp, LIM);
    const float TiltAngle = LIM[index.combined()].TiltAngle;
    // float P = R;
    // .. Scaling factor of drift in 2D approximation.  to account Neutral sheet
    float fth = 0; /* scaling function of drift vel */
    float Dftheta_dtheta = 0;
    const float TiltPos_r = qp.r;
    const float TiltPos_th = Pi / 2.f - TiltAngle;
    const float TiltPos_phi = qp.phi;
    float Vsw = SolarWindSpeed(index, {TiltPos_r, TiltPos_th, TiltPos_phi}, LIM);
    // const float dthetans = fabsf(GeV / (c * aum) * (2. * r * R) / (Asun * sqrtf(
    //                                                                    1 + Gamma_Bfield(r, TiltPos_th, Vsw) *
    //                                                                    Gamma_Bfield(r, TiltPos_th, Vsw) + (
    //                                                                        IsPolarRegion
    //                                                                            ? delta_Bfield(r, TiltPos_th) *
    //                                                                                delta_Bfield(
    //                                                                                    r, TiltPos_th)
    //                                                                            : 0)))); /*dTheta_ns = 2*R_larmor/r*/
    const float dthetans = fabsf(GeV / (SoL * aum) * (2.f * qp.r * qp.R) / (Asun * sqrtf(
                                                                                1 + sq(Gamma_Bfield(
                                                                                    qp.r, TiltPos_th, Vsw)) +
                                                                                IsPolarRegion * sq(
                                                                                    delta_Bfield(qp.r, TiltPos_th)))));


    //               (pt.A*sqrt(Ek*(Ek+2*pt.T0)))/fabs(pt.Z)                                   B_mag_alfa    = Asun/r2*sqrt(1.+ Gamma_alfa*Gamma_alfa + delta_alfa*delta_alfa );
    // double B2_mag_alfa   = B_mag_alfa*B_mag_alfa;
    //       dthetans = fabs((GeV/(c*aum))*(2.*       (MassNumber*sqrt(T*(T+2*T0))         ))/(Z*r*sqrt(B2_mag_alfa)));  /*dTheta_ns = 2*R_larmor/r*/


    if (const float theta_mez = Pi / 2.f - 0.5f * sinf(fminf(TiltAngle + dthetans, Pi / 2.f));
        theta_mez < Pi / .2f) {
        const float a_f = acosf(Pi / (2.f * theta_mez) - 1);
        fth = 1.f / a_f * atanf((1.f - 2.f * qp.th / Pi) * tanf(a_f));
        Dftheta_dtheta = -2.f * tanf(a_f) / (a_f * Pi * (
                                                 1.f + (1 - 2.f * qp.th / Pi) * (1 - 2.f * qp.th / Pi) * tanf(a_f) *
                                                 tanf(a_f)));
    } else {
        /* if the smoothness parameter "theta_mez" is greater then Pi/2, then the neutral sheet is flat, only heaviside function is applied.*/
        fth = sign(qp.th - Pi / 2.f);
    }
    // if (debug){
    //    printf("Vsw(tilt) %e\tBMagAlpha=%e\tAsun=%e\tr2=%e\tGamma_alfa=%e\tdelta_alfa=%e\n\n", Vsw,(Asun/(r*r))*sqrt( 1+Gamma_Bfield(r,TiltPos_th,Vsw)*Gamma_Bfield(r,TiltPos_th,Vsw)+delta_Bfield(r,TiltPos_th)*delta_Bfield(r,TiltPos_th)),
    //             Asun,r*r, Gamma_Bfield(r,TiltPos_th,Vsw), ((IsPolarRegion)?delta_Bfield(r,TiltPos_th)*delta_Bfield(r,TiltPos_th):0));
    //    printf("KA %f\tdthetans=%e\tftheta=%e\tDftheta_dtheta=%e\n", Ka, dthetans,fth,Dftheta_dtheta);
    // }
    Vsw = SolarWindSpeed(index, qp, LIM); // solar wind evaluated
    // .. drift velocity

    const float dm = IsPolarRegion ? delta_m : 0;

    const float E = dm * dm * qp.r * qp.r * Vsw * Vsw + Omega * Omega * rhelio * rhelio * (qp.r - rhelio) * (
                        qp.r - rhelio)
                    * sinf(qp.th) * sinf(qp.th) * sinf(qp.th) * sinf(qp.th) + rhelio * rhelio * Vsw * Vsw * sinf(qp.th)
                    * sinf(qp.th);
    float C = fth * sinf(qp.th) * Ka * qp.r * rhelio / (Asun * E * E);
    C *= qp.R * qp.R / (qp.R * qp.R + LIM[index.combined()].P0d * LIM[index.combined()].P0d);
    /* drift reduction factor. <------------------------------ */
    //reg drift
    v.r = -C * Omega * rhelio * 2 * (qp.r - rhelio) * sinf(qp.th) * (
              (2 * dm * dm * qp.r * qp.r + rhelio * rhelio * sinf(qp.th) * sinf(qp.th)) * Vsw * Vsw * Vsw * cosf(qp.th)
              -
              .5f * (dm * dm * qp.r * qp.r * Vsw * Vsw - Omega * Omega * rhelio * rhelio * (qp.r - rhelio) * (
                         qp.r - rhelio) * sinf(qp.th) * sinf(qp.th) * sinf(qp.th) * sinf(qp.th) + rhelio * rhelio * Vsw
                     * Vsw *
                     sinf(qp.th)
                     * sinf(qp.th)) * sinf(qp.th) * dV_dth);
    v.th = -C * Omega * rhelio * Vsw * sinf(qp.th) * sinf(qp.th) * (
               2 * qp.r * (qp.r - rhelio) * (
                   dm * dm * qp.r * Vsw * Vsw + Omega * Omega * rhelio * rhelio * (qp.r - rhelio) * sinf(qp.th) *
                   sinf(qp.th) * sinf(qp.th) * sinf(qp.th)) - (4 * qp.r - 3 * rhelio) * E);
    v.phi = 2 * C * Vsw * (-dm * dm * qp.r * qp.r * (dm * qp.r + rhelio * cosf(qp.th)) * Vsw * Vsw * Vsw + 2 *
                           dm * qp.r * E * Vsw - Omega * Omega * rhelio * rhelio * (qp.r - rhelio) * sinf(qp.th) *
                           sinf(qp.th) * sinf(qp.th) * sinf(qp.th) * (
                               dm * qp.r * qp.r * Vsw - rhelio * (qp.r - rhelio) * Vsw * cosf(qp.th) + rhelio * (
                                   qp.r - rhelio)
                               * sinf(qp.th) * dV_dth));
    //ns drift
    C = Vsw * Dftheta_dtheta * sinf(qp.th) * sinf(qp.th) * Ka * qp.r * rhelio * rhelio / (Asun * E);
    C *= qp.R * qp.R / (qp.R * qp.R + LIM[index.combined()].P0dNS * LIM[index.combined()].P0dNS);
    /* drift reduction factor.  <------------------------------ */
    v.r += -C * Omega * sinf(qp.th) * (qp.r - rhelio);
    //v.th += 0;
    v.phi += -C * Vsw;

    // Suppression rigidity dependence: logistic function (~1 at low energy ---> ~0 at high energy)
    const float HighRigiSupp = LIM[index.combined()].plateau + (1.f - LIM[index.combined()].plateau) / (
                                   1.f + expf(HighRigiSupp_smoothness * (qp.R - HighRigiSupp_TransPoint)));
    v.r *= HighRigiSupp;
    v.th *= HighRigiSupp;
    v.phi *= HighRigiSupp;

    return v;
}


float EvalP0DriftSuppressionFactor(const int WhichDrift, const int SolarPhase, const float TiltAngleDeg,
                                   const float ssn) {
    //WhichDrift = 0 - Regular drift
    //WhichDrift = 1 - NS drift

    float InitialVal, FinalVal, CenterOfTransition, smoothness;

    if (WhichDrift == 0) {
        // reg drift
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
        //NS drift
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
    ////////////////////////////////////////////////
    // printf("-- SolarPhase_v[izone]: %d \n",SolarPhase);
    // printf("-- TiltAngle_v[izone] : %.0lf \n",TiltAngleDeg);
    // if (WhichDrift==0)
    // {
    //   printf("-- TDDS_P0d_Ini       : %e \n",InitialVal);
    //   printf("-- TDDS_P0d_Fin       : %.e \n",FinalVal);
    //   printf("-- CenterOfTransition : %e \n",CenterOfTransition);
    //   printf("-- smoothness         : %e \n",smoothness);
    //   printf("-- P0d                : %e \n",SmoothTransition(InitialVal,  FinalVal,  CenterOfTransition,  smoothness,  TiltAngleDeg));


    // }
    ////////////////////////////////////////////////
    return SmoothTransition(InitialVal, FinalVal, CenterOfTransition, smoothness, TiltAngleDeg);
}

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
