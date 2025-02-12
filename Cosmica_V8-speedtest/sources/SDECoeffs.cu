#include <cmath>
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"
#include "GenComputation.cuh"
#include "SolarWind.cuh"
#include "MagneticDrift.cuh"
#include "DiffusionModel.cuh"
#include "SDECoeffs.cuh"

__device__ DiffusionTensor_t trivial_DiffusionTensor_symmetric(int ZoneNum) {
    const DiffusionTensor_t KSym;

    return KSym;
}

__device__ Tensor3D_t trivial_SquareRoot_DiffusionTensor(DiffusionTensor_t KSym) {
    Tensor3D_t Ddif;
    Ddif.rr = 1;

    return Ddif;
}

__device__ vect3D_t trivial_AdvectiveTerm(DiffusionTensor_t KSym) {
    const vect3D_t AdvTerm;

    return AdvTerm;
}

__device__ float trivial_EnergyLoss() {
    const float R_loss = 0.1;
    // float R_loss = 0;

    return R_loss;
}


////////////////////////////////////////////////////////////////
//.....  Diffusion Term  .....................................
////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
//..... Simmetric component and derivative  ....................
////////////////////////////////////////////////////////////////

__device__ DiffusionTensor_t DiffusionTensor_symmetric(const unsigned int InitZone, const signed int HZone,
                                                       const float r, const float th, const float phi, const float R,
                                                       const PartDescription_t pt, const float GaussRndNumber) {
    /*Authors: 2022 Stefano */
    /* * description: Evaluate the symmetric component (and derivative) of diffusion tensor in heliocentric coordinates
        \param InitZone initial Zone in the heliosphere (in the list of parameters)
        \param HZone   Zone in the Heliosphere
        \param part    particle position and energy
        \return  the symmetric component (and derivative) of diffusion tensor
     */
    DiffusionTensor_t KK;
    if (HZone < Heliosphere.Nregions) {
        /*  NOTE about HMF
         *  In the equatorial region, we used the Parker’s IMF (B Par ) in the parametrization of Hattingh and Burger (1995),
         *  while in the polar regions we used a modiﬁed IMF (B Pol ) that includes a latitudinal component,
         *  accounting for large scale ﬂuctuations, dominant at high heliolatitudes, as suggested by Jokipii and Kota (1989).
         *  The polar region is defined by the constant 'PolarZone' and 'CosPolarZone' (globals.h)
         *  Note that since in equatorial region the theta component of HMF is zero we implemented two cases to avoid 0calculations.
         */
        const bool IsPolarRegion = fabsf(cosf(th)) > CosPolarZone;
        const float SignAsun = LIM[HZone + InitZone].Asun > 0 ? +1. : -1.;
        // ....... Get Diffusion tensor in HMF frame
        // Kpar = Kh.x    dKpar_dr = dK_dr.x
        // Kp1  = Kh.y    dKp1_dr  = dK_dr.y
        // Kp2  = Kh.z    dKp2_dr  = dK_dr.z
        float3 dK_dr;
        const auto [Kpar, Kperp, Kperp2] =
                Diffusion_Tensor_In_HMF_Frame(InitZone, HZone, r, th, beta_R(R, pt), R, GaussRndNumber, dK_dr);


        // ....... Define the HMF Model
        // The implemented model is the Parker’s IMF with Jokipii and Kota (1989) modification in polar regions
        // if (fmod(fabs(th)-M_PI, M_PI)<1e-6) th = 1e-6;   // Correction for divergent Bth in polar regions

        const float PolSign = th - Pi / 2. > 0 ? -1. : +1.;
        const float DelDirac = th == Pi / 2. ? 1. : 0.;
        const float V_SW = SolarWindSpeed(InitZone, HZone, r, th, phi);

        const float Br = PolSign; // divided by A/r^2
        const float Bth = IsPolarRegion ? r * delta_m / (rhelio * sinf(th)) : 0.f; // divided by A/r^2
        const float Bph = -PolSign * (Omega * (r - rhelio) * sinf(th) / V_SW); // divided by A/r^2

        const float dBr_dr = -2.f * PolSign / r;
        const float dBr_dth = -2.f * DelDirac;

        const float dBth_dr = IsPolarRegion ? -delta_m / (rhelio * sinf(th)) : 0.f;
        const float dBth_dth = IsPolarRegion ? r * delta_m / (rhelio * sinf(th) * sinf(th)) * -cosf(th) : 0.f;

        const float dBph_dr = PolSign * (r - 2.f * rhelio) * (Omega * sinf(th)) / (r * V_SW);
        const float dBph_dth = -(r - rhelio) * Omega * (
                                   -PolSign * (cosf(th) * V_SW - sinf(th) *
                                               DerivativeOfSolarWindSpeed_dtheta(InitZone, HZone, r, th, phi)) + 2.f *
                                   sinf(th) * V_SW * DelDirac) / (V_SW * V_SW);

        const float HMF_Mag2 = 1 + Bth * Bth + Bph * Bph;
        const float HMF_Mag = sqrtf(HMF_Mag2);
        const float dBMag_dr = (Br * dBr_dr + Bth * dBth_dr + Bph * dBph_dr) / HMF_Mag;
        const float dBMag_dth = (Br * dBr_dth + Bth * dBth_dth + Bph * dBph_dth) / HMF_Mag;

        const float sqrtBR2BT2 = sqrtf(sq(Br) + sq(Bth)); //sqrt(Br^2+Bth^2)
        const float dsqrtBR2BT2_dr = (Br * dBr_dr + Bth * dBth_dr) / sqrtBR2BT2;
        const float dsqrtBR2BT2_dth = (Br * dBr_dth + Bth * dBth_dth) / sqrtBR2BT2;


        const float sinPsi = SignAsun * (-Bph / HMF_Mag);
        const float cosPsi = sqrtBR2BT2 / HMF_Mag;
        const float sinZeta = SignAsun * (Bth / sqrtBR2BT2);
        const float cosZeta = SignAsun * (Br / sqrtBR2BT2); //Br/sqrt(Br^2+Bth^2)

        // float sinPsi  = sinPsi  *sinPsi  ;
        // float cosPsi  = cosPsi  *cosPsi  ;
        // float sinZeta2 = sinZeta *sinZeta ;
        // float cosZeta = cosZeta *cosZeta ;

        const float DsinPsi_dr = -SignAsun * (dBph_dr * HMF_Mag - Bph * dBMag_dr) / HMF_Mag2;
        const float DsinPsi_dtheta = -SignAsun * (dBph_dth * HMF_Mag - Bph * dBMag_dth) / HMF_Mag2; //


        // float DcosPsi_dr     = ( dsqrtBR2BT2_dr *HMF_Mag - sqrtBR2BT2 * dBMag_dr )/HMF_Mag2;
        // float DcosPsi_dtheta = ( dsqrtBR2BT2_dth*HMF_Mag - sqrtBR2BT2 * dBMag_dth)/HMF_Mag2;//
        const float DcosPsi_dr = Bph * (-Br * Br * dBph_dr + Bph * Br * dBr_dr + Bth * (-Bth * dBph_dr + Bph * dBth_dr))
                                 / (
                                     sqrtBR2BT2 * HMF_Mag2 * HMF_Mag);
        const float DcosPsi_dtheta = Bph * (-Br * Br * dBph_dth + Bph * Br * dBr_dth + Bth * (
                                                -Bth * dBph_dth + Bph * dBth_dth)) / (sqrtBR2BT2 * HMF_Mag2 * HMF_Mag);


        const float DsinZeta_dr = SignAsun * (dBth_dr * sqrtBR2BT2 - Bth * dsqrtBR2BT2_dr) / (sqrtBR2BT2 * sqrtBR2BT2);
        const float DsinZeta_dtheta = SignAsun * (dBth_dth * sqrtBR2BT2 - Bth * dsqrtBR2BT2_dth) / (
                                          sqrtBR2BT2 * sqrtBR2BT2);
        const float DcosZeta_dr = SignAsun * (dBr_dr * sqrtBR2BT2 - Br * dsqrtBR2BT2_dr) / (sqrtBR2BT2 * sqrtBR2BT2);
        const float DcosZeta_dtheta = SignAsun * (dBr_dth * sqrtBR2BT2 - Br * dsqrtBR2BT2_dth) / (
                                          sqrtBR2BT2 * sqrtBR2BT2);

        // ....... rotate Diffusion tensor from HMF to heliocentric frame
        /* The complete calculations of diffusion tensor in helioscentric frame are the follow :
         * note: in this case the diff tens is [Kpar,Kper2,Kper3]
         * KK.rr = Kp2*sq(sinZeta) + sq(cosZeta) * (Kpar*sq(cosPsi) + Kp3*sq(sinPsi));
         * KK.tt = Kp2*sq(cosZeta) + sq(sinZeta) * (Kpar*sq(cosPsi) + Kp3*sq(sinPsi));
         * KK.pp = Kpar*sq(sinPsi) + Kp3*sq(cosPsi);
         * KK.rt = sinZeta*cosZeta*(Kpar*sq(cosPsi)+Kp3*sq(sinPsi)-Kp2);
         * KK.tr = sinZeta*cosZeta*(Kpar*sq(cosPsi)+Kp3*sq(sinPsi)-Kp2);
         * KK.rp = -(Kpar-Kp3) *sinPsi*cosPsi*cosZeta;
         * KK.pr = -(Kpar-Kp3) *sinPsi*cosPsi*cosZeta;
         * KK.tp = -(Kpar-Kp3) *sinPsi*cosPsi*sinZeta;
         * KK.pt = -(Kpar-Kp3) *sinPsi*cosPsi*sinZeta;
         */
#define POLAR_BRANCH_REDUCE
#ifndef POLAR_BRANCH_REDUCE
        if (IsPolarRegion) {
        // polar region
#endif
        KK.rr = Kperp * sq(sinZeta) + sq(cosZeta) * (Kpar * sq(cosPsi) + Kperp2 * sq(sinPsi));
        KK.tt = Kperp * sq(cosZeta) + sq(sinZeta) * (Kpar * sq(cosPsi) + Kperp2 * sq(sinPsi));
        KK.pp = Kpar * sq(sinPsi) + Kperp2 * sq(cosPsi);
        KK.tr = sinZeta * cosZeta * (Kpar * sq(cosPsi) + Kperp2 * sq(sinPsi) - Kperp);
        KK.pr = -(Kpar - Kperp2) * sinPsi * cosPsi * cosZeta;
        KK.pt = -(Kpar - Kperp2) * sinPsi * cosPsi * sinZeta;
#ifndef POLAR_BRANCH_REDUCE
        } else {
            // equatorial region. Bth = 0
            // --> sinZeta = 0
            // --> cosZeta = +/- 1
            KK.rr = +sq(cosZeta) * (Kh.x * sq(cosPsi) + Kh.z * sq(sinPsi));
            KK.tt = Kh.y * sq(cosZeta);
            KK.pp = Kh.x * sq(sinPsi) + Kh.z * sq(cosPsi);
            KK.tr = 0.;
            KK.pr = -(Kh.x - Kh.z) * sinPsi * cosPsi * cosZeta;
            KK.pt = 0.;
        }
#endif

        // ....... evaluate derivative of diffusion tensor
        /*  The complete calculations of derivatives are the follow:
         * note: in this case the diff tens is [Kpar,Kper2,Kper3]
         *
         *  KK.DKrr_dr = 2. * cosZeta*(sq(cosPsi)*Kpar+Kp3*sq(sinPsi))*DcosZeta_dr+sq(sinZeta)*DKp2_dr+sq(cosZeta)*(2. * cosPsi*Kpar*DcosPsi_dr+sq(cosPsi)*DKpar_dr+sinPsi* (sinPsi*DKp3_dr+2.*Kp3*DsinPsi_dr )) + 2.*Kp2*sinZeta*DsinZeta_dr;
         *  KK.DKtt_dt = 2. * cosZeta*Kp2*DcosZeta_dtheta+sq(cosZeta) * DKp2_dtheta + sq(sinZeta) * (2.*cosPsi*Kpar*DcosPsi_dtheta+sq(cosPsi)*DKpar_dtheta+sinPsi*(sinPsi*DKp3_dtheta+2.*Kp3*DsinPsi_dtheta))+2.*(sq(cosPsi)*Kpar+Kp3*sq(sinPsi))*sinZeta*DsinZeta_dtheta;
         *  KK.DKpp_dp = 2.*cosPsi*Kp3*DcosPsi_dphi + sq(cosPsi) * DKp3_dphi + sinPsi*(sinPsi*DKpar_dphi+2. *Kpar * DsinPsi_dphi);
         *  KK.DKrt_dr = (-Kp2+sq(cosPsi)*Kpar+Kp3*sq(sinPsi))*sinZeta*DcosZeta_dr    +cosZeta*sinZeta*(2.*cosPsi*Kpar*DcosPsi_dr     - DKp2_dr    +sq(cosPsi)*DKpar_dr    +sinPsi*(sinPsi*DKp3_dr    +2.*Kp3*DsinPsi_dr    ))+cosZeta*(-Kp2+sq(cosPsi)*Kpar+Kp3*sq(sinPsi))*DsinZeta_dr    ;
         *  KK.DKtr_dt = (-Kp2+sq(cosPsi)*Kpar+Kp3*sq(sinPsi))*sinZeta*DcosZeta_dtheta+cosZeta*sinZeta*(2.*cosPsi*Kpar*DcosPsi_dtheta - DKp2_dtheta+sq(cosPsi)*DKpar_dtheta+sinPsi*(sinPsi*DKp3_dtheta+2.*Kp3*DsinPsi_dtheta))+cosZeta*(-Kp2+sq(cosPsi)*Kpar+Kp3*sq(sinPsi))*DsinZeta_dtheta;
         *  KK.DKrp_dr = cosZeta * (Kp3-Kpar) * sinPsi * DcosPsi_dr + cosPsi * (Kp3-Kpar) *sinPsi * DcosZeta_dr + cosPsi* cosZeta* sinPsi* (DKp3_dr-DKpar_dr) + cosPsi*cosZeta*(Kp3-Kpar) * DsinPsi_dr;
         *  KK.DKpr_dp = cosPsi*( Kp3 - Kpar ) *sinPsi * DcosZeta_dphi + cosZeta * ( sinPsi * ((Kp3-Kpar) * DcosPsi_dphi + cosPsi* (DKp3_dphi-DKpar_dphi)) + cosPsi* (Kp3-Kpar) * DsinPsi_dphi );
         *  KK.DKtp_dt = (Kp3 - Kpar ) *sinPsi*sinZeta*DcosPsi_dtheta+cosPsi*sinPsi*sinZeta*(DKp3_dtheta-DKpar_dtheta)+cosPsi*(Kp3-Kpar)*sinZeta*DsinPsi_dtheta+cosPsi*(Kp3-Kpar)*sinPsi*DsinZeta_dtheta;
         *  KK.DKpt_dp = sinZeta * ( sinPsi * ( ( Kp3 - Kpar)*DcosPsi_dphi+cosPsi*(DKp3_dphi-DKpar_dphi ) ) + cosPsi*( Kp3 - Kpar ) *DsinPsi_dphi ) + cosPsi*( Kp3-Kpar )*sinPsi*DsinZeta_dphi ;
         *
         */
        // Here we apply some semplification due to HMF and Kdiff description
        // B field do not depend on phi
        // Kpar,Kperp1-2 do not depends on theta and phi
#ifndef POLAR_BRANCH_REDUCE
        if (IsPolarRegion) {
        // polar region
#endif
        KK.DKrr_dr = 2.f * cosZeta * (sq(cosPsi) * Kpar + Kperp2 * sq(sinPsi)) * DcosZeta_dr + sinZeta *
                     sinZeta * dK_dr.y + sq(cosZeta) * (
                         2.f * cosPsi * Kpar * DcosPsi_dr + sq(cosPsi) * dK_dr.x + sinPsi * (
                             sinPsi * dK_dr.z + 2.f * Kperp2 * DsinPsi_dr)) + 2.f * Kperp * sinZeta * DsinZeta_dr;
        KK.DKtt_dt = 2.f * cosZeta * Kperp * DcosZeta_dtheta + sq(sinZeta) * (
                         2.f * cosPsi * Kpar * DcosPsi_dtheta + 2.f * sinPsi * Kperp2 * DsinPsi_dtheta) + 2.f * (
                         sq(cosPsi) * Kpar + Kperp2 * sq(sinPsi)) * sinZeta * DsinZeta_dtheta;
        // KK.DKpp_dp = 0. ;
        KK.DKrt_dr = (-Kperp + sq(cosPsi) * Kpar + Kperp2 * sq(sinPsi)) * (
                         sinZeta * DcosZeta_dr + cosZeta * DsinZeta_dr) + cosZeta * sinZeta * (
                         2.f * cosPsi * Kpar * DcosPsi_dr + sq(cosPsi) * dK_dr.x - dK_dr.y + sinPsi * (
                             sinPsi * dK_dr.z + 2.f * Kperp2 * DsinPsi_dr));
        KK.DKtr_dt = (-Kperp + sq(cosPsi) * Kpar + Kperp2 * sq(sinPsi)) * (
                         sinZeta * DcosZeta_dtheta + cosZeta * DsinZeta_dtheta) + cosZeta * sinZeta * (
                         2.f * cosPsi * Kpar * DcosPsi_dtheta + 2.f * sinPsi * Kperp2 * DsinPsi_dtheta);
        KK.DKrp_dr = cosZeta * (Kperp2 - Kpar) * sinPsi * DcosPsi_dr + cosPsi * (Kperp2 - Kpar) * sinPsi * DcosZeta_dr +
                     cosPsi * cosZeta * sinPsi * (dK_dr.z - dK_dr.x) + cosPsi * cosZeta * (Kperp2 - Kpar) *
                     DsinPsi_dr; //-----------qui
        // KK.DKpr_dp = 0. ;
        KK.DKtp_dt = (Kperp2 - Kpar) * (sinPsi * sinZeta * DcosPsi_dtheta + cosPsi * (
                                            sinZeta * DsinPsi_dtheta + sinPsi * DsinZeta_dtheta));
        // KK.DKpt_dp = 0. ;
#ifndef POLAR_BRANCH_REDUCE
        } else {
            // equatorial region. Bth = 0
            // --> sinZeta = 0
            // --> cosZeta = 1
            // --> derivative of sinZeta or cosZeta = 0
            KK.DKrr_dr = 2.f * cosZeta * (sq(cosPsi) * Kh.x + Kh.z * sq(sinPsi)) * DcosZeta_dr + cosZeta *
                         cosZeta * (2.f * cosPsi * Kh.x * DcosPsi_dr + sq(cosPsi) * dK_dr.x + sinPsi * (
                                        sinPsi * dK_dr.z + 2.f * Kh.z * DsinPsi_dr));
            KK.DKtt_dt = 2.f * cosZeta * Kh.y * DcosZeta_dtheta;
            // KK.DKpp_dp = 0. ;
            KK.DKrt_dr = 0.;
            KK.DKtr_dt = 0.;
            KK.DKrp_dr = cosZeta * (Kh.z - Kh.x) * sinPsi * DcosPsi_dr + cosPsi * (Kh.z - Kh.x) * sinPsi * DcosZeta_dr +
                         cosPsi * cosZeta * sinPsi * (dK_dr.z - dK_dr.x) + cosPsi * cosZeta * (Kh.z - Kh.x) *
                         DsinPsi_dr; //-----------qui
            // KK.DKpr_dp = 0. ;
            KK.DKtp_dt = 0.;
            // KK.DKpt_dp = 0. ;
        }
#endif
    } else {
        // heliosheat ...............................
        KK.rr = Diffusion_Coeff_heliosheat(InitZone, r, th, phi, beta_R(R, pt), R, KK.DKrr_dr);
    }
    return KK;
}

///////////////////////////////////////////////////////
// ========== Decomposition of Diffution Tensor ======
// solve the equation (square root of diffusion tensor)
//  | a b c |   | g h i | | g l o |
//  | b d e | = | l m n | | h m p |
//  | c e f |   | o p q | | i n q |
//
// the diffusion tensor is written in the form that arise from SDE calculation
///////////////////////////////////////////////////////
__device__ Tensor3D_t SquareRoot_DiffusionTerm(const signed int HZone, DiffusionTensor_t K, const float r,
                                               const float th, int *res) {
    /* * description:  ========== Decomposition of Diffusion Tensor ======
                      solve square root of diffusion tensor in heliocentric spherical coordinates
        \par Tensor3D K     input tensor
        \par qvect_t  part  particle Position
        \par int      &res  0 if ok; 1 if error
       */

    // // ---- Create diffusion matrix of FPE from diffusion tensor
    Tensor3D_t D;

    K.rr = 2.f * K.rr;
    D.rr = sqrtf(K.rr); // g = sqrt(a)
    if (HZone < Heliosphere.Nregions) {
        //K.rt=2.*K.rt/r;                       K.rp=2.*K.rp/(r*sinf(th));
        K.tr = 2.f * K.tr / r;
        K.tt = 2.f * K.tt / (r * r); //K.tp=2.*K.tp/(sq(r)*sinf(th));
        K.pr = 2.f * K.pr / (r * sinf(th));
        K.pt = 2.f * K.pt / (r * r * sinf(th));
        K.pp = 2.f * K.pp / (r * r * sinf(th) * sinf(th));

        // ---- square root of diffusion tensor in heliocentric spherical coordinates
        // // first try
        // // h=0 i=0 n=0
        // // D.rt = 0;
        // // D.rp = 0;
        // // D.tp = 0;
        // do this only in the inner heliosphere (questo perchè D.pp vale nan nell'outer heliosphere, forse legato alla radice di un elemento negativo (i.e. arrotondamenti allo zero non ottimali))
        D.tr = K.tr / D.rr; // l = b/g
        D.pr = K.pr / D.rr; // o = c/g
        D.tt = sqrtf(K.tt - D.tr * D.tr); // m = sqrt(d-l^2)
        D.pt = 1.f / D.tt * (K.pt - D.tr * D.pr); // p = 1/m (e-lo)
        D.pp = sqrtf(K.pp - D.pr * D.pr - D.pt * D.pt); // q = sqrt(f - o^2 -p^2)
    }

    // check if ok
    if (const float sum = D.rr + D.tr + D.pr + D.tt + D.pt + D.pp; isnan(sum) || isinf(sum)) {
        // there was some error...
        // -- TODO -- check an other solution... see Pei et al 2010 or Kopp et al 2012
        // not implemented since such cases are rare
        *res = 1;
    } else {
        *res = 0;
    }

    return D;
}

////////////////////////////////////////////////////////////////
//.....  Advective term of SDE  ................................
////////////////////////////////////////////////////////////////

// -- Radial --------------------------------------------------
// dr_Adv = 2.* K.rr/r + K.DKrr_dr + K.tr/(r*tanf(th)) + K.DKtr_dt/r + K.DKpr_dp/(r*sinf(th)) ;
// dr_Adv+= - Vsw - vdr - vdns  ;
// -- latitudinal ----------------------------------------------
// dtheta_Adv = K.rt/(sq(r)) + K.tt/(tanf(th)*sq(r) ) + K.DKrt_dr/r + K.DKtt_dt/sq(r) + K.DKpt_dp / (sinf(th)*sq(r));
//_Adv+= - vdth/r;
// -- Azimutal -------------------------------------------------
// dphi_Adv = K.rp/(sq(r)*sin(theta))+ K.DKrp_dr/(r*sinf(th)) + K.DKtp_dt/( sq(r)*sinf(th)) + K.DKpp_dp/( sq(r)*sq(sinf(th))) ;
// dphi_Adv+= - (vdph+vdns_p)/(r*sinf(th));
///////////////////////////
__device__ vect3D_t AdvectiveTerm(const unsigned int InitZone, const signed int HZone, const DiffusionTensor_t &K,
                                  const float r, const float th, const float phi, const float R,
                                  const PartDescription_t pt) {
    vect3D_t AdvTerm = {2.f * K.rr / r + K.DKrr_dr, 0, 0};

    if (HZone < Heliosphere.Nregions) {
        // inner Heliosphere .........................
        // advective part related to diffision tensor
        AdvTerm.r += K.tr / (r * tanf(th)) + K.DKtr_dt / r;
        AdvTerm.th += K.tr / sq(r) + K.tt / (tanf(th) * sq(r)) + K.DKrt_dr / r + K.DKtt_dt / sq(r);

        AdvTerm.phi += K.pr / (sq(r) * sinf(th)) + K.DKrp_dr / (r * sinf(th)) + K.DKtp_dt / (sq(r) * sinf(th));
        // drift component
        const auto [drift_r, drift_th, drift_phi] = Drift_PM89(InitZone, HZone, r, th, phi, R, pt);
        AdvTerm.r -= drift_r;
        AdvTerm.th -= drift_th / r;
        AdvTerm.phi -= drift_phi / (r * sinf(th));
    }

    // convective part related to solar wind
    AdvTerm.r -= SolarWindSpeed(InitZone, HZone, r, th, phi);

    return AdvTerm;
}

////////////////////////////////////////////////////////////////
//..... Energy loss term   .....................................
////////////////////////////////////////////////////////////////
__device__ float EnergyLoss(const unsigned int InitZone, const signed int HZone, const float r, const float th,
                            const float phi, const float R) {
    if (HZone < Heliosphere.Nregions) {
        // inner Heliosphere .........................
        return 2.f / 3.f * SolarWindSpeed(InitZone, HZone, r, th, phi) / r * R;
        // (Ek + 2.*T0)/(Ek + T0) * Ek = pt.Z*pt.Z/(pt.A*pt.A)*sq(R)/(sqrt(pt.Z*pt.Z/(pt.A*pt.A)*sq(R) + pt.T0*pt.T0))
    }
    // heliosheat ...............................
    // no energy loss
    return 0;
}


////////////////////////////////////////////////////////////////
//..... Loss term  .............................................
////////////////////////////////////////////////////////////////
/* __device__ float LossTerm(unsigned char InitZone,signed char HZone, float r, float th, float phi, float Ek, float T0) {
   // xxx da completare
    if (HZone<Heliosphere.Nregions) {
        // inner Heliosphere .........................
        return 2.*SolarWindSpeed(InitZone,HZone, r, th, phi)/r *( 1./3.*( sq(Ek)+2*Ek*T0+ 2*sq(T0) )/( (Ek+T0)*(Ek+T0) ) -1 );
                                                                      // ( sq(Ek)+2*Ek*T0+ 2*sq(T0) )/( (Ek+T0)*(Ek+T0) ) -1 ) = -T*T0/((T+T0)*(T+T0))
    }

    else {
        // heliosheat ...............................
        // solar wind divergence is zero
        return 0;
    }

    return 1;
} */
