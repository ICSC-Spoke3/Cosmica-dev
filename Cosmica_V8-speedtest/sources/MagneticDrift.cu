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

__device__ vect3D_t Drift_PM89(const unsigned int InitZone, const signed int HZone, const float r, const float th,
                               const float phi, const float R, const PartDescription_t pt) {
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
    const float IsPolarRegion = fabsf(cosf(th)) > CosPolarZone;
    const float Ka = safeSign(pt.Z) * GeV * beta_R(R, pt) * R / 3;
    /* sign(Z)*beta*P/3 constant part of antisymmetric diffusion coefficient */
    // pt.A*sqrt(Ek*(Ek+2*pt.T0))/pt.Z
    const float Asun = LIM[HZone + InitZone].Asun; /* Magnetic Field Amplitude constant / aum^2*/
    const float dV_dth = DerivativeOfSolarWindSpeed_dtheta(InitZone, HZone, r, th, phi);
    const float TiltAngle = LIM[HZone + InitZone].TiltAngle;
    // float P = R;
    // .. Scaling factor of drift in 2D approximation.  to account Neutral sheet
    float fth = 0; /* scaling function of drift vel */
    float Dftheta_dtheta = 0;
    const float TiltPos_r = r;
    const float TiltPos_th = Pi / 2. - TiltAngle;
    const float TiltPos_phi = phi;
    float Vsw = SolarWindSpeed(InitZone, HZone, TiltPos_r, TiltPos_th, TiltPos_phi);
    // const float dthetans = fabsf(GeV / (c * aum) * (2. * r * R) / (Asun * sqrtf(
    //                                                                    1 + Gamma_Bfield(r, TiltPos_th, Vsw) *
    //                                                                    Gamma_Bfield(r, TiltPos_th, Vsw) + (
    //                                                                        IsPolarRegion
    //                                                                            ? delta_Bfield(r, TiltPos_th) *
    //                                                                                delta_Bfield(
    //                                                                                    r, TiltPos_th)
    //                                                                            : 0)))); /*dTheta_ns = 2*R_larmor/r*/
    const float dthetans = fabsf(GeV / (c * aum) * (2.f * r * R) / (Asun * sqrtf(
                                                                        1 + sq(Gamma_Bfield(r, TiltPos_th, Vsw)) +
                                                                        IsPolarRegion * sq(
                                                                            delta_Bfield(r, TiltPos_th)))));


    //               (pt.A*sqrt(Ek*(Ek+2*pt.T0)))/fabs(pt.Z)                                   B_mag_alfa    = Asun/r2*sqrt(1.+ Gamma_alfa*Gamma_alfa + delta_alfa*delta_alfa );
    // double B2_mag_alfa   = B_mag_alfa*B_mag_alfa;
    //       dthetans = fabs((GeV/(c*aum))*(2.*       (MassNumber*sqrt(T*(T+2*T0))         ))/(Z*r*sqrt(B2_mag_alfa)));  /*dTheta_ns = 2*R_larmor/r*/


    if (const float theta_mez = Pi / 2. - 0.5 * sinf(fminf(TiltAngle + dthetans, Pi / 2.));
        theta_mez < Pi / .2) {
        const float a_f = acosf(Pi / (2. * theta_mez) - 1);
        fth = 1.f / a_f * atanf((1.f - 2.f * th / Pi) * tanf(a_f));
        Dftheta_dtheta = -2.f * tanf(a_f) / (a_f * Pi * (
                                                 1.f + (1 - 2.f * th / Pi) * (1 - 2.f * th / Pi) * tanf(a_f) *
                                                 tanf(a_f)));
    } else {
        /* if the smoothness parameter "theta_mez" is greater then Pi/2, then the neutral sheet is flat, only heaviside function is applied.*/
        fth = sign(th - Pi / 2.f);
    }
    // if (debug){
    //    printf("Vsw(tilt) %e\tBMagAlpha=%e\tAsun=%e\tr2=%e\tGamma_alfa=%e\tdelta_alfa=%e\n\n", Vsw,(Asun/(r*r))*sqrt( 1+Gamma_Bfield(r,TiltPos_th,Vsw)*Gamma_Bfield(r,TiltPos_th,Vsw)+delta_Bfield(r,TiltPos_th)*delta_Bfield(r,TiltPos_th)),
    //             Asun,r*r, Gamma_Bfield(r,TiltPos_th,Vsw), ((IsPolarRegion)?delta_Bfield(r,TiltPos_th)*delta_Bfield(r,TiltPos_th):0));
    //    printf("KA %f\tdthetans=%e\tftheta=%e\tDftheta_dtheta=%e\n", Ka, dthetans,fth,Dftheta_dtheta);
    // }
c    Vsw = SolarWindSpeed(InitZone, HZone, r, th, phi); // solar wind evaluated
    // .. drift velocity

#ifndef POLAR_BRANCH_REDUCE
    if (IsPolarRegion) {
        // polar region
#endif
        const float E = delta_m * delta_m * r * r * Vsw * Vsw + Omega * Omega * rhelio * rhelio * (r - rhelio) * (
                            r - rhelio)
                        * sinf(th) * sinf(th) * sinf(th) * sinf(th) + rhelio * rhelio * Vsw * Vsw * sinf(th) * sinf(th);
        float C = fth * sinf(th) * Ka * r * rhelio / (Asun * E * E);
        C *= R * R / (R * R + LIM[HZone + InitZone].P0d * LIM[HZone + InitZone].P0d);
        /* drift reduction factor. <------------------------------ */
        //reg drift
        v.r = -C * Omega * rhelio * 2 * (r - rhelio) * sinf(th) * (
                  (2 * delta_m * delta_m * r * r + rhelio * rhelio * sinf(th) * sinf(th)) * Vsw * Vsw * Vsw * cosf(th) -
                  .5f * (delta_m * delta_m * r * r * Vsw * Vsw - Omega * Omega * rhelio * rhelio * (r - rhelio) * (
                             r - rhelio) * sinf(th) * sinf(th) * sinf(th) * sinf(th) + rhelio * rhelio * Vsw * Vsw *
                         sinf(th)
                         * sinf(th)) * sinf(th) * dV_dth);
        v.th = -C * Omega * rhelio * Vsw * sinf(th) * sinf(th) * (
                   2 * r * (r - rhelio) * (
                       delta_m * delta_m * r * Vsw * Vsw + Omega * Omega * rhelio * rhelio * (r - rhelio) * sinf(th) *
                       sinf(th) * sinf(th) * sinf(th)) - (4 * r - 3 * rhelio) * E);
        v.phi = 2 * C * Vsw * (-delta_m * delta_m * r * r * (delta_m * r + rhelio * cosf(th)) * Vsw * Vsw * Vsw + 2 *
                               delta_m * r * E * Vsw - Omega * Omega * rhelio * rhelio * (r - rhelio) * sinf(th) *
                               sinf(th) * sinf(th) * sinf(th) * (
                                   delta_m * r * r * Vsw - rhelio * (r - rhelio) * Vsw * cosf(th) + rhelio * (
                                       r - rhelio)
                                   * sinf(th) * dV_dth));
        //ns drift
        C = Vsw * Dftheta_dtheta * sinf(th) * sinf(th) * Ka * r * rhelio * rhelio / (Asun * E);
        C *= R * R / (R * R + LIM[HZone + InitZone].P0dNS * LIM[HZone + InitZone].P0dNS);
        /* drift reduction factor.  <------------------------------ */
        v.r += -C * Omega * sinf(th) * (r - rhelio);
        //v.th += 0;
        v.phi += -C * Vsw;
#ifndef POLAR_BRANCH_REDUCE
    } else {
        // equatorial region. Bth = 0
        const float E = +Omega * Omega * (r - rhelio) * (r - rhelio) * sinf(th) * sinf(th) + Vsw * Vsw;
        float C = Omega * fth * Ka * r / (Asun * E * E);
        C *= R * R / (R * R + LIM[HZone + InitZone].P0d * LIM[HZone + InitZone].P0d);
        /* drift reduction factor.  <------------------------------ */
        v.r = -2.f * C * (r - rhelio) * (
                  .5f * (Omega * Omega * (r - rhelio) * (r - rhelio) * sinf(th) * sinf(th) - Vsw * Vsw) * sinf(th) *
                  dV_dth + Vsw * Vsw * Vsw * cosf(th));
        v.th = -C * Vsw * sinf(th) * (2 * Omega * Omega * r * (r - rhelio) * (r - rhelio) * sinf(th) * sinf(th) - (
                                          4 * r - 3.f * rhelio) * E);
        // float Gamma=(Omega*(r-rhelio)*sinf(th))/Vsw;
        // float Gamma2=Gamma*Gamma;
        // float DGamma_dr     = Omega*sinf(th)/Vsw;
        // v.th = (Ka *r*fth*(Gamma*(3.+3.*Gamma2)+r*(1.-Gamma2)*DGamma_dr))/(Asun*(1.+Gamma2)*(1.+Gamma2));


        v.phi = 2 * C * Vsw * Omega * (r - rhelio) * (r - rhelio) * sinf(th) * (Vsw * cosf(th) - sinf(th) * dV_dth);
        // if (debug){
        //    printf("Vdr %e  vdth %e vfph %e  \n", v.r,v.th,v.phi);
        // }
        C = Vsw * Dftheta_dtheta * Ka * r / (Asun * E);
        C *= R * R / (R * R + LIM[HZone + InitZone].P0dNS * LIM[HZone + InitZone].P0dNS);
        /* drift reduction factor.  <------------------------------ */
        v.r += -C * Omega * (r - rhelio) * sinf(th);
        v.phi += -C * Vsw;
        // if (debug){
        //    printf("VdNSr %e  VdNSph %e  \n", - C*Omega*(r-rhelio)*sinf(th),  - C*Vsw);
        //    printf("Drift_PM89:: Dftheta_dtheta=%e\t KA=%e\tr=%f\tAsun=%e\tGamma2=%e\n",Dftheta_dtheta,Ka,r,Asun,Vsw*Vsw*Vsw*Vsw/(E*E ));
        // }
    }
#endif

    // Suppression rigidity dependence: logistic function (~1 at low energy ---> ~0 at high energy)
    const float HighRigiSupp = LIM[HZone + InitZone].plateau + (1.f - LIM[HZone + InitZone].plateau) / (
                                   1.f + expf(HighRigiSupp_smoothness * (R - HighRigiSupp_TransPoint)));
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
