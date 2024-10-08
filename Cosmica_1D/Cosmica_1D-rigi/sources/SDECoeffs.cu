#include <math.h>
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"
#include "GenComputation.cuh"
#include "SolarWind.cuh"
#include "MagneticDrift.cuh"
#include "DiffusionModel.cuh"
#include "SDECoeffs.cuh"

__device__ struct DiffusionTensor_t trivial_DiffusionTensor_symmetric(){
  struct DiffusionTensor_t KSym;

  return KSym;
}

__device__ struct Tensor3D_t trivial_SquareRoot_DiffusionTensor(float K0, float R, PartDescription_t pt){ // int InitZone, int HZone, float T, struct PartDescription_t pt
  struct Tensor3D_t Ddif;

  // int high_activity = Heliosphere.IsHighActivityPeriod[InitZone]?0:1;
  // float K0 = LIM[HZone+InitZone].k0_paral[high_activity];

  // if (HZone<Heliosphere.Nregions) Ddif.rr = K0*beta_R(R, pt)*R;

  // else Ddif.rr = HS[HZone].k0*beta_R(R, pt)*R;

  Ddif.rr = K0*beta_R(R, pt)*R;

  return Ddif;
}

__device__ struct vect3D_t trivial_AdvectiveTerm(struct Tensor3D_t Ddif, float V0, float r){ // struct Tensor3D_t Ddif, int InitZone, int HZone, float r
  struct vect3D_t AdvTerm;
  // float V0=(HZone<Heliosphere.Nregions)?LIM[HZone+InitZone].V0:HS[InitZone].V0;

  AdvTerm.r = 2.*Ddif.rr/r - V0;

  return AdvTerm;
}

__device__ float trivial_EnergyLoss(float R, float r, float V0, struct PartDescription_t pt){ // int InitZone, int HZone, float T, float r, struct PartDescription_t pt

  // float V0 = (HZone<Heliosphere.Nregions)?LIM[HZone+InitZone].V0:HS[InitZone].V0;
  
  // if (HZone<Heliosphere.Nregions) return 2.*V0*(T+2.*pt.T0)/(T+pt.T0)*T/(3.*r);

  // else return 0.;

  return 2.*V0*R/(3.0*r);
}


////////////////////////////////////////////////////////////////
//.....  Diffusion Term  .....................................
////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
//..... Simmetric component and derivative  ....................
////////////////////////////////////////////////////////////////

__device__ struct DiffusionTensor_t DiffusionTensor_symmetric(unsigned char InitZone, signed char HZone, float r, float th, float phi, float R, struct PartDescription_t pt, float GaussRndNumber){
  /*Authors: 2022 Stefano */ 
  /* * description: Evaluate the symmetric component (and derivative) of diffusion tensor in heliocentric coordinates
      \param InitZone initial Zone in the heliosphere (in the list of parameters)
      \param HZone   Zone in the Heliosphere
      \param part    particle position and energy
      \return  the symmetric component (and derivative) of diffusion tensor
   */
  struct DiffusionTensor_t KK;
  if (HZone<Heliosphere.Nregions){
    /*  NOTE about HMF
     *  In the equatorial region, we used the Parker’s IMF (B Par ) in the parametrization of Hattingh and Burger (1995), 
     *  while in the polar regions we used a modiﬁed IMF (B Pol ) that includes a latitudinal component, 
     *  accounting for large scale ﬂuctuations, dominant at high heliolatitudes, as suggested by Jokipii and Kota (1989).
     *  The polar region is defined by the constant 'PolarZone' and 'CosPolarZone' (globals.h)
     *  Note that since in equatorial region the theta component of HMF is zero we implemented two cases to avoid 0calculations.
     */
    bool IsPolarRegion = (fabs(cos(th))>CosPolarZone )? true:false;
    float SignAsun = (LIM[HZone+InitZone].Asun>0)?+1.:-1.;
    // ....... Get Diffusion tensor in HMF frame
    // Kpar = Kh.x    dKpar_dr = dK_dr.x  
    // Kp1  = Kh.y    dKp1_dr  = dK_dr.y
    // Kp2  = Kh.z    dKp2_dr  = dK_dr.z
    float3 dK_dr;
    float3 Kh = Diffusion_Tensor_In_HMF_Frame(InitZone,HZone,r,th,beta_R(R,pt),R,GaussRndNumber,dK_dr);

    
    // ....... Define the HMF Model
    // The implemented model is the Parker’s IMF with Jokipii and Kota (1989) modification in polar regions
    // if (fmod(fabs(th)-M_PI, M_PI)<1e-6) th = 1e-6;   // Correction for divergent Bth in polar regions

    int PolSign = (th-Pi/2.>0)? -1.:+1.;
    int DelDirac= (th==Pi/2.)?1:0;
    float V_SW  = SolarWindSpeed(InitZone, HZone, r, th, phi);

    float Br = float(PolSign);                                                // divided by A/r^2
    float Bth= (IsPolarRegion)?r*delta_m/(rhelio*sin(th)):0.;       // divided by A/r^2
    float Bph= - float(PolSign)*((Omega*(r-rhelio)*sin(th))/V_SW);  // divided by A/r^2

    float dBr_dr   = -2.*PolSign/(r);
    float dBr_dth  = -2.*float(DelDirac);

    float dBth_dr  = (IsPolarRegion)?-delta_m/(rhelio*sin(th)):0.;
    float dBth_dth = (IsPolarRegion)?r*delta_m/(rhelio*sin(th)*sin(th))*(-cos(th)):0.;

    float dBph_dr  = float(PolSign) *(r-2.*rhelio)*(Omega*sin(th))/(r*V_SW);
    float dBph_dth = - (r-rhelio)*Omega* ( -PolSign*( cos(th) *V_SW - sin(th)*DerivativeOfSolarWindSpeed_dtheta(InitZone, HZone, r, th, phi) ) + 2. * sin(th) *V_SW*float(DelDirac)) /(V_SW*V_SW);
         
    float HMF_Mag2   = 1 + Bth*Bth + Bph*Bph; 
    float HMF_Mag    = sqrt(HMF_Mag2);        
    float dBMag_dr   = (Br*dBr_dr  + Bth*dBth_dr  + Bph*dBph_dr )/HMF_Mag;
    float dBMag_dth  = (Br*dBr_dth + Bth*dBth_dth + Bph*dBph_dth)/HMF_Mag;

    float sqrtBR2BT2 =  sqrt( Br*Br + Bth*Bth); //sqrt(Br^2+Bth^2)
    float dsqrtBR2BT2_dr   = (Br*dBr_dr  + Bth*dBth_dr  )/sqrtBR2BT2 ;
    float dsqrtBR2BT2_dth  = (Br*dBr_dth + Bth*dBth_dth )/sqrtBR2BT2;


    float sinPsi  = SignAsun*(         - Bph / HMF_Mag);
    float cosPsi  =          (   sqrtBR2BT2  / HMF_Mag);
    float sinZeta = SignAsun*(          Bth  / sqrtBR2BT2);
    float cosZeta = SignAsun*(            Br / sqrtBR2BT2); //Br/sqrt(Br^2+Bth^2)

    // float sinPsi  = sinPsi  *sinPsi  ;
    // float cosPsi  = cosPsi  *cosPsi  ;
    // float sinZeta2 = sinZeta *sinZeta ;
    // float cosZeta = cosZeta *cosZeta ;
    
    float DsinPsi_dr     = -SignAsun*( dBph_dr *HMF_Mag - Bph * dBMag_dr )/HMF_Mag2;
    float DsinPsi_dtheta = -SignAsun*( dBph_dth*HMF_Mag - Bph * dBMag_dth)/HMF_Mag2;//


    // float DcosPsi_dr     = ( dsqrtBR2BT2_dr *HMF_Mag - sqrtBR2BT2 * dBMag_dr )/HMF_Mag2;
    // float DcosPsi_dtheta = ( dsqrtBR2BT2_dth*HMF_Mag - sqrtBR2BT2 * dBMag_dth)/HMF_Mag2;//
    float DcosPsi_dr     = (Bph*(-Br*Br * dBph_dr  + Bph * Br * dBr_dr  + Bth*(-Bth*dBph_dr  + Bph * dBth_dr )))/(sqrtBR2BT2 *HMF_Mag2*HMF_Mag );
    float DcosPsi_dtheta = (Bph*(-Br*Br * dBph_dth + Bph * Br * dBr_dth + Bth*(-Bth*dBph_dth + Bph * dBth_dth)))/(sqrtBR2BT2 *HMF_Mag2*HMF_Mag  );
   
    
    float DsinZeta_dr     = SignAsun*( dBth_dr *sqrtBR2BT2 - Bth * dsqrtBR2BT2_dr )/(sqrtBR2BT2*sqrtBR2BT2);
    float DsinZeta_dtheta = SignAsun*( dBth_dth*sqrtBR2BT2 - Bth * dsqrtBR2BT2_dth)/(sqrtBR2BT2*sqrtBR2BT2);
    float DcosZeta_dr     = SignAsun*( dBr_dr *sqrtBR2BT2  - Br  * dsqrtBR2BT2_dr )/(sqrtBR2BT2*sqrtBR2BT2);
    float DcosZeta_dtheta = SignAsun*( dBr_dth*sqrtBR2BT2  - Br * dsqrtBR2BT2_dth)/(sqrtBR2BT2*sqrtBR2BT2);
    
    // ....... rotate Diffusion tensor from HMF to heliocentric frame
    /* The complete calculations of diffusion tensor in helioscentric frame are the follow :
     * note: in this case the diff tens is [Kpar,Kper2,Kper3]
     * KK.rr = Kp2*sinZeta*sinZeta + cosZeta*cosZeta * (Kpar*cosPsi*cosPsi + Kp3*sinPsi*sinPsi);
     * KK.tt = Kp2*cosZeta*cosZeta + sinZeta*sinZeta * (Kpar*cosPsi*cosPsi + Kp3*sinPsi*sinPsi);
     * KK.pp = Kpar*sinPsi*sinPsi + Kp3*cosPsi*cosPsi;
     * KK.rt = sinZeta*cosZeta*(Kpar*cosPsi*cosPsi+Kp3*sinPsi*sinPsi-Kp2);
     * KK.tr = sinZeta*cosZeta*(Kpar*cosPsi*cosPsi+Kp3*sinPsi*sinPsi-Kp2);
     * KK.rp = -(Kpar-Kp3) *sinPsi*cosPsi*cosZeta;
     * KK.pr = -(Kpar-Kp3) *sinPsi*cosPsi*cosZeta;
     * KK.tp = -(Kpar-Kp3) *sinPsi*cosPsi*sinZeta;
     * KK.pt = -(Kpar-Kp3) *sinPsi*cosPsi*sinZeta;
     */
    
    if (IsPolarRegion){
      // polar region
      KK.rr = Kh.y*sinZeta*sinZeta + cosZeta*cosZeta * (Kh.x*cosPsi*cosPsi + Kh.z*sinPsi*sinPsi);
      KK.tt = Kh.y*cosZeta*cosZeta + sinZeta*sinZeta * (Kh.x*cosPsi*cosPsi + Kh.z*sinPsi*sinPsi);
      KK.pp = Kh.x*sinPsi*sinPsi + Kh.z*cosPsi*cosPsi;
      KK.tr = sinZeta*cosZeta*(Kh.x*cosPsi*cosPsi+Kh.z*sinPsi*sinPsi-Kh.y);
      KK.pr = -(Kh.x-Kh.z) *sinPsi*cosPsi*cosZeta;
      KK.pt = -(Kh.x-Kh.z) *sinPsi*cosPsi*sinZeta;
    }else{
      // equatorial region. Bth = 0 
      // --> sinZeta = 0 
      // --> cosZeta = +/- 1
      KK.rr =               + cosZeta*cosZeta * (Kh.x*cosPsi*cosPsi + Kh.z*sinPsi*sinPsi);
      KK.tt = Kh.y*cosZeta*cosZeta ;
      KK.pp = Kh.x*sinPsi*sinPsi + Kh.z*cosPsi*cosPsi;
      KK.tr = 0.;
      KK.pr = -(Kh.x-Kh.z) *sinPsi*cosPsi*cosZeta;
      KK.pt = 0.;
    }

    // ....... evaluate derivative of diffusion tensor
    /*  The complete calculations of derivatives are the follow:
     * note: in this case the diff tens is [Kpar,Kper2,Kper3]
     *
     *  KK.DKrr_dr = 2. * cosZeta*(cosPsi*cosPsi*Kpar+Kp3*sinPsi*sinPsi)*DcosZeta_dr+sinZeta*sinZeta*DKp2_dr+cosZeta*cosZeta*(2. * cosPsi*Kpar*DcosPsi_dr+cosPsi*cosPsi*DKpar_dr+sinPsi* (sinPsi*DKp3_dr+2.*Kp3*DsinPsi_dr )) + 2.*Kp2*sinZeta*DsinZeta_dr;
     *  KK.DKtt_dt = 2. * cosZeta*Kp2*DcosZeta_dtheta+cosZeta*cosZeta * DKp2_dtheta + sinZeta*sinZeta * (2.*cosPsi*Kpar*DcosPsi_dtheta+cosPsi*cosPsi*DKpar_dtheta+sinPsi*(sinPsi*DKp3_dtheta+2.*Kp3*DsinPsi_dtheta))+2.*(cosPsi*cosPsi*Kpar+Kp3*sinPsi*sinPsi)*sinZeta*DsinZeta_dtheta;
     *  KK.DKpp_dp = 2.*cosPsi*Kp3*DcosPsi_dphi + cosPsi*cosPsi * DKp3_dphi + sinPsi*(sinPsi*DKpar_dphi+2. *Kpar * DsinPsi_dphi);
     *  KK.DKrt_dr = (-Kp2+cosPsi*cosPsi*Kpar+Kp3*sinPsi*sinPsi)*sinZeta*DcosZeta_dr    +cosZeta*sinZeta*(2.*cosPsi*Kpar*DcosPsi_dr     - DKp2_dr    +cosPsi*cosPsi*DKpar_dr    +sinPsi*(sinPsi*DKp3_dr    +2.*Kp3*DsinPsi_dr    ))+cosZeta*(-Kp2+cosPsi*cosPsi*Kpar+Kp3*sinPsi*sinPsi)*DsinZeta_dr    ;
     *  KK.DKtr_dt = (-Kp2+cosPsi*cosPsi*Kpar+Kp3*sinPsi*sinPsi)*sinZeta*DcosZeta_dtheta+cosZeta*sinZeta*(2.*cosPsi*Kpar*DcosPsi_dtheta - DKp2_dtheta+cosPsi*cosPsi*DKpar_dtheta+sinPsi*(sinPsi*DKp3_dtheta+2.*Kp3*DsinPsi_dtheta))+cosZeta*(-Kp2+cosPsi*cosPsi*Kpar+Kp3*sinPsi*sinPsi)*DsinZeta_dtheta;
     *  KK.DKrp_dr = cosZeta * (Kp3-Kpar) * sinPsi * DcosPsi_dr + cosPsi * (Kp3-Kpar) *sinPsi * DcosZeta_dr + cosPsi* cosZeta* sinPsi* (DKp3_dr-DKpar_dr) + cosPsi*cosZeta*(Kp3-Kpar) * DsinPsi_dr;
     *  KK.DKpr_dp = cosPsi*( Kp3 - Kpar ) *sinPsi * DcosZeta_dphi + cosZeta * ( sinPsi * ((Kp3-Kpar) * DcosPsi_dphi + cosPsi* (DKp3_dphi-DKpar_dphi)) + cosPsi* (Kp3-Kpar) * DsinPsi_dphi );
     *  KK.DKtp_dt = (Kp3 - Kpar ) *sinPsi*sinZeta*DcosPsi_dtheta+cosPsi*sinPsi*sinZeta*(DKp3_dtheta-DKpar_dtheta)+cosPsi*(Kp3-Kpar)*sinZeta*DsinPsi_dtheta+cosPsi*(Kp3-Kpar)*sinPsi*DsinZeta_dtheta;
     *  KK.DKpt_dp = sinZeta * ( sinPsi * ( ( Kp3 - Kpar)*DcosPsi_dphi+cosPsi*(DKp3_dphi-DKpar_dphi ) ) + cosPsi*( Kp3 - Kpar ) *DsinPsi_dphi ) + cosPsi*( Kp3-Kpar )*sinPsi*DsinZeta_dphi ;
     *  
     */
    // Here we apply some semplification due to HMF and Kdiff description
    // B field do not depend on phi
    // Kpar,Kperp1-2 do not depends on theta and phi
    if (IsPolarRegion){
      // polar region
      KK.DKrr_dr = 2. * cosZeta*(cosPsi*cosPsi*Kh.x+Kh.z*sinPsi*sinPsi)*DcosZeta_dr + sinZeta*sinZeta*dK_dr.y + cosZeta*cosZeta*(2. * cosPsi*Kh.x*DcosPsi_dr+cosPsi*cosPsi*dK_dr.x+ sinPsi* (sinPsi*dK_dr.z+2.*Kh.z*DsinPsi_dr )) + 2.*Kh.y*sinZeta*DsinZeta_dr;
      KK.DKtt_dt = 2. * cosZeta*Kh.y*DcosZeta_dtheta + sinZeta*sinZeta * (2.*cosPsi*Kh.x*DcosPsi_dtheta+2.*sinPsi*Kh.z*DsinPsi_dtheta)+2.*(cosPsi*cosPsi*Kh.x+Kh.z*sinPsi*sinPsi)*sinZeta*DsinZeta_dtheta;
      // KK.DKpp_dp = 0. ;
      KK.DKrt_dr = (-Kh.y+cosPsi*cosPsi*Kh.x+Kh.z*sinPsi*sinPsi)*(sinZeta*DcosZeta_dr    +cosZeta*DsinZeta_dr     )+cosZeta*sinZeta*(2.*cosPsi*Kh.x*DcosPsi_dr     +cosPsi*cosPsi*dK_dr.x - dK_dr.y +sinPsi*(sinPsi*dK_dr.z    +2.*Kh.z*DsinPsi_dr    ));
      KK.DKtr_dt = (-Kh.y+cosPsi*cosPsi*Kh.x+Kh.z*sinPsi*sinPsi)*(sinZeta*DcosZeta_dtheta+cosZeta*DsinZeta_dtheta )+cosZeta*sinZeta*(2.*cosPsi*Kh.x*DcosPsi_dtheta                             +2.*sinPsi*Kh.z*DsinPsi_dtheta)                  ;
      KK.DKrp_dr = cosZeta * (Kh.z-Kh.x) * sinPsi * DcosPsi_dr    + cosPsi  * (Kh.z-Kh.x) *sinPsi * DcosZeta_dr + cosPsi* cosZeta* sinPsi* (dK_dr.z-dK_dr.x) + cosPsi * cosZeta*(Kh.z-Kh.x) * DsinPsi_dr;//-----------qui
      // KK.DKpr_dp = 0. ;
      KK.DKtp_dt = (Kh.z - Kh.x) * (sinPsi*sinZeta*DcosPsi_dtheta + cosPsi*(sinZeta*DsinPsi_dtheta + sinPsi*DsinZeta_dtheta) );
      // KK.DKpt_dp = 0. ;
    }else{
      // equatorial region. Bth = 0 
      // --> sinZeta = 0 
      // --> cosZeta = 1
      // --> derivative of sinZeta or cosZeta = 0
      KK.DKrr_dr = 2. * cosZeta*(cosPsi*cosPsi*Kh.x+Kh.z*sinPsi*sinPsi)*DcosZeta_dr +  cosZeta*cosZeta*(2. * cosPsi*Kh.x*DcosPsi_dr+cosPsi*cosPsi*dK_dr.x+ sinPsi* (sinPsi*dK_dr.z+2.*Kh.z*DsinPsi_dr )) ;
      KK.DKtt_dt = 2. * cosZeta*Kh.y*DcosZeta_dtheta ;
      // KK.DKpp_dp = 0. ;
      KK.DKrt_dr = 0.;
      KK.DKtr_dt = 0.;
      KK.DKrp_dr = cosZeta * (Kh.z-Kh.x) * sinPsi * DcosPsi_dr    + cosPsi  * (Kh.z-Kh.x) *sinPsi * DcosZeta_dr + cosPsi* cosZeta* sinPsi* (dK_dr.z-dK_dr.x) + cosPsi * cosZeta*(Kh.z-Kh.x) * DsinPsi_dr;//-----------qui
      // KK.DKpr_dp = 0. ;
      KK.DKtp_dt = 0.;
      // KK.DKpt_dp = 0. ;
    }
 
  }else{
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
__device__ struct Tensor3D_t SquareRoot_DiffusionTerm(signed char HZone, struct DiffusionTensor_t K, float r, float th, int* res){
/* * description:  ========== Decomposition of Diffusion Tensor ====== 
                  solve square root of diffusion tensor in heliocentric spherical coordinates
    \par Tensor3D K     input tensor
    \par qvect_t  part  particle Position
    \par int      &res  0 if ok; 1 if error
   */

    // // ---- Create diffusion matrix of FPE from diffusion tensor
    float sintheta=sin(th);
    K.rr=2.*K.rr;                    
    if (HZone<Heliosphere.Nregions) {
                                        //K.rt=2.*K.rt/r;                       K.rp=2.*K.rp/(r*sintheta);
        K.tr=2.*K.tr/r;              K.tt=2.*K.tt/(r*r);           //K.tp=2.*K.tp/(r2*sintheta);
        K.pr=2.*K.pr/(r*sintheta);   K.pt=2.*K.pt/(r*r*sintheta);  K.pp=2.*K.pp/(r*r*sintheta*sintheta);     
    }

    // ---- square root of diffusion tensor in heliocentric spherical coordinates
    struct Tensor3D_t D;
    // // first try
    // // h=0 i=0 n=0
    // // D.rt = 0;
    // // D.rp = 0;
    // // D.tp = 0;
    D.rr = sqrt(K.rr);                    // g = sqrt(a)
    if (HZone<Heliosphere.Nregions) {      // do this only in the inner heliosphere (questo perchè D.pp vale nan nell'outer heliosphere, forse legato alla radice di un elemento negativo (i.e. arrotondamenti allo zero non ottimali))
        D.tr = K.tr/D.rr;                     // l = b/g
        D.pr = K.pr/D.rr;                     // o = c/g
        D.tt = sqrt(K.tt - D.tr*D.tr);        // m = sqrt(d-l^2)
        D.pt = 1./D.tt * ( K.pt - D.tr*D.pr );// p = 1/m (e-lo)
        D.pp = sqrt( K.pp - D.pr*D.pr - D.pt*D.pt); // q = sqrt(f - o^2 -p^2)
    }

    // check if ok
    if ((isinf(D.rr))||(isinf(D.tr))||(isinf(D.pr))||(isinf(D.tt))||(isinf(D.pt))||(isinf(D.pp))
    ||(isnan(D.rr))||(isnan(D.tr))||(isnan(D.pr))||(isnan(D.tt))||(isnan(D.pt))||(isnan(D.pp)) ){
        // there was some error...
        // -- TODO -- check an other solution... see Pei et al 2010 or Kopp et al 2012
        // not implemented since such cases are rare
        *res = 1;
    }
    else {
        *res = 0;
    }

    return D;
}

   ////////////////////////////////////////////////////////////////
   //.....  Advective term of SDE  ................................
   ////////////////////////////////////////////////////////////////

   // -- Radial --------------------------------------------------
   // dr_Adv = 2.* K.rr/r + K.DKrr_dr + K.tr/(r*tantheta) + K.DKtr_dt/r + K.DKpr_dp/(r*sintheta) ;
   // dr_Adv+= - Vsw - vdr - vdns  ;
   // -- latitudinal ----------------------------------------------
   // dtheta_Adv = K.rt/(r2) + K.tt/(tantheta*r2 ) + K.DKrt_dr/r + K.DKtt_dt/r2 + K.DKpt_dp / (sintheta*r2);
   //_Adv+= - vdth/r;
   // -- Azimutal -------------------------------------------------
   // dphi_Adv = K.rp/(r2*sin(theta))+ K.DKrp_dr/(r*sintheta) + K.DKtp_dt/( r2*sintheta) + K.DKpp_dp/( r2*sintheta*sintheta) ;
   // dphi_Adv+= - (vdph+vdns_p)/(r*sintheta);
   ///////////////////////////
__device__ struct vect3D_t AdvectiveTerm(unsigned char InitZone,signed char HZone, struct DiffusionTensor_t K, float r, float th, float phi, float R, struct PartDescription_t pt){
    struct vect3D_t AdvTerm;
    if (HZone<Heliosphere.Nregions) {
        // inner Heliosphere .........................
        float sintheta=sin(th);
        float tantheta=tan(th);
        float r2      = r*r;
        // advective part related to diffision tensor
        AdvTerm.r   = 2.* K.rr/r + K.DKrr_dr + K.tr/(r*tantheta) + K.DKtr_dt/r ;
        AdvTerm.th  = K.tr/(r2) + K.tt/(tantheta*r2 ) + K.DKrt_dr/r + K.DKtt_dt/r2 ;       

        AdvTerm.phi = K.pr/(r2*sintheta)+ K.DKrp_dr/(r*sintheta) + K.DKtp_dt/( r2*sintheta)  ;
        // drift component
        struct vect3D_t v_drift = Drift_PM89(InitZone, HZone, r, th, phi, R, pt);
        AdvTerm.r   += - v_drift.r  ;
        AdvTerm.th  += - v_drift.th/r ;
        AdvTerm.phi += - v_drift.phi/(r*sintheta);

    }
    
    else {
        // heliosheat ...............................
        // advective part related to diffision tensor
        AdvTerm.r   = 2.* K.rr/r + K.DKrr_dr ;
        AdvTerm.th  = 0;       
        AdvTerm.phi = 0;
        // drift component     
        // -- none --
    }

    // convective part related to solar wind
    AdvTerm.r  += -SolarWindSpeed(InitZone, HZone, r, th, phi);

    return AdvTerm;
}

  ////////////////////////////////////////////////////////////////
  //..... Energy loss term   .....................................
  ////////////////////////////////////////////////////////////////
__device__ float EnergyLoss(unsigned char InitZone, signed char HZone, float r, float th, float phi, float R) {
    if (HZone<Heliosphere.Nregions) {
        // inner Heliosphere .........................
        return 2./3.*SolarWindSpeed(InitZone, HZone, r, th, phi)/r * R;
        // (Ek + 2.*T0)/(Ek + T0) * Ek = pt.Z*pt.Z/(pt.A*pt.A)*R*R/(sqrt(pt.Z*pt.Z/(pt.A*pt.A)*R*R + pt.T0*pt.T0))
    }
    
    else {
        // heliosheat ...............................
        // no energy loss
        return 0;
    }
}


  ////////////////////////////////////////////////////////////////
  //..... Loss term  .............................................
  ////////////////////////////////////////////////////////////////
/* __device__ float LossTerm(unsigned char InitZone,signed char HZone, float r, float th, float phi, float Ek, float T0) {
   // xxx da completare
    if (HZone<Heliosphere.Nregions) {
        // inner Heliosphere .........................
        return 2.*SolarWindSpeed(InitZone,HZone, r, th, phi)/r *( 1./3.*( Ek*Ek+2*Ek*T0+ 2*T0*T0 )/( (Ek+T0)*(Ek+T0) ) -1 );
                                                                      // ( Ek*Ek+2*Ek*T0+ 2*T0*T0 )/( (Ek+T0)*(Ek+T0) ) -1 ) = -T*T0/((T+T0)*(T+T0))
    }
    
    else {
        // heliosheat ...............................
        // solar wind divergence is zero
        return 0;
    }

    return 1;   
} */