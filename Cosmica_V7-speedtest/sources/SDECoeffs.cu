#include <math.h>
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"
#include "GenComputation.cuh"
#include "SolarWind.cuh"
#include "MagneticDrift.cuh"
#include "DiffusionModel.cuh"
#include "SDECoeffs.cuh"

__device__ struct DiffusionTensor_t trivial_DiffusionTensor_symmetric(int ZoneNum){
    struct DiffusionTensor_t KSym;

    return KSym;
}

__device__ struct Tensor3D_t trivial_SquareRoot_DiffusionTensor(struct DiffusionTensor_t KSym){
    struct Tensor3D_t Ddif;
    Ddif.rr = 1;

    return Ddif;
}

__device__ struct vect3D_t trivial_AdvectiveTerm(struct DiffusionTensor_t KSym){
    struct vect3D_t AdvTerm;

    return AdvTerm;
}

__device__ float trivial_EnergyLoss(){
    float R_loss = 0.1;
    // float R_loss = 0;

    return R_loss;
}


////////////////////////////////////////////////////////////////
//.....  Diffusion Term  .....................................
////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
//..... Coordinate transformations  ............................
////////////////////////////////////////////////////////////////

__device__ float eval_Bth(float r, float th, int IsPolarRegion) {

  return (IsPolarRegion)?r*delta_m/(rhelio*sin(th)):0.;       // divided by A/r^2
}

__device__ float eval_Bph(float r, float th, float PolSign, float V_SW) {

  return - float(PolSign)*((Omega*(r-rhelio)*sin(th))/V_SW);  // divided by A/r^2
}

__device__ float eval_HMF_Mag(float r, float th, bool IsPolarRegion, float Bth, float Bph) {

  return sqrt(1 + Bth*Bth + Bph*Bph);
}

__device__ float eval_sqrtBR2BT2(float PolSign, float Bth) {

  return sqrt(PolSign*PolSign + Bth*Bth); //sqrt(Br^2+Bth^2)
}

__device__ float eval_sinPsi(float SignAsun, float Bph, float HMF_Mag) {
  
  return SignAsun*(- Bph/HMF_Mag);
}

__device__ float eval_cosPsi(float PolSign, float Bth, float HMF_Mag) {

  return (sqrt(PolSign*PolSign + Bth*Bth)/HMF_Mag);
}

__device__ float eval_sinZeta(float PolSign, float SignAsun, float Bth) {

  return SignAsun*(Bth/sqrt(PolSign*PolSign + Bth*Bth));
}

__device__ float eval_cosZeta(float PolSign, float SignAsun, float Bth) {

  return SignAsun*(PolSign/sqrt(PolSign*PolSign + Bth*Bth)); //Br/sqrt(Br^2+Bth^2)
}

// Derivatives //

__device__ float eval_dBth_dr(float IsPolarRegion, float th) {

  return (IsPolarRegion)?-delta_m/(rhelio*sin(th)):0.;
}

__device__ float eval_dBph_dr(float r, float th, float PolSign, float V_SW) {
  
  return float(PolSign) *(r-2.*rhelio)*(Omega*sin(th))/(r*V_SW);
}

__device__ float eval_dBth_dth(float IsPolarRegion, float r, float th) {

  return (IsPolarRegion)?r*delta_m/(rhelio*sin(th)*sin(th))*(-cos(th)):0.;
}

__device__ float eval_dBph_dth(float r, float th, float PolSign, float V_SW, float dV_SWdth, float DelDirac) {

  return - (r-rhelio)*Omega* ( -PolSign*( cos(th) *V_SW - sin(th)*dV_SWdth ) + 2. * sin(th) *V_SW*DelDirac) /(V_SW*V_SW);
}

__device__ float eval_dBMag_dth(float PolSign, float dV_SWdth, float DelDirac, float Bth, float dBth_dth, float Bph, float dBph_dth, float HMF_Mag) {

  return (PolSign*(-2.*DelDirac) + Bth*dBth_dth + Bph*dBph_dth)/HMF_Mag;
}

__device__ float eval_dBMag_dr(float r, float PolSign, float Bth, float Bph, float dBth_dr, float dBph_dr, float HMF_Mag) {

  return (PolSign*(-2.*PolSign/(r))  + Bth*dBth_dr  + Bph*dBph_dr )/HMF_Mag;
}

__device__ float eval_DsinPsi_dr(float SignAsun, float HMF_Mag, float Bph, float dBph_dr, float dBMag_dr) {

  return -SignAsun*( dBph_dr *HMF_Mag - Bph * dBMag_dr )/(HMF_Mag*HMF_Mag);
}

__device__ float eval_DsinPsi_dtheta(float SignAsun, float HMF_Mag, float Bph, float dBph_dth, float dBMag_dth) {

  return -SignAsun*( dBph_dth*HMF_Mag - Bph * dBMag_dth)/(HMF_Mag*HMF_Mag);
}

__device__ float eval_dsqrtBR2BT2_dr(float r, float PolSign, float Bth, float sqrtBR2BT2, float dBth_dr) {

  return (PolSign*(-2.*PolSign/(r))  + Bth*dBth_dr  )/sqrtBR2BT2 ;
}

__device__ float eval_dsqrtBR2BT2_dth(float PolSign, float Bth, float sqrtBR2BT2, float dBth_dth, float DelDirac) {

  return (PolSign*(-2.*DelDirac) + Bth*dBth_dth )/sqrtBR2BT2;
}

__device__ float eval_DcosPsi_dr(float r, float PolSign, float Bth, float Bph, float sqrtBR2BT2, float dBth_dr, float dBph_dr, float HMF_Mag) {

  return (Bph*(-PolSign*PolSign*dBph_dr + Bph*PolSign*(-2.*PolSign/(r)) + Bth*(-Bth*dBph_dr + Bph*dBth_dr )))/(sqrtBR2BT2*(HMF_Mag*HMF_Mag)*HMF_Mag);
}

__device__ float eval_DcosPsi_dtheta(float PolSign, float Bth, float Bph, float sqrtBR2BT2, float dBph_dth, float dBth_dth, float HMF_Mag, float DelDirac) {

  return (Bph*(-PolSign*PolSign*dBph_dth + Bph*PolSign*(-2.*DelDirac) + Bth*(-Bth*dBph_dth + Bph*dBth_dth)))/(sqrtBR2BT2*(HMF_Mag*HMF_Mag)*HMF_Mag);
}

__device__ float eval_DsinZeta_dr(float SignAsun, float Bth, float dBth_dr, float sqrtBR2BT2, float dsqrtBR2BT2_dr) {

  return SignAsun*( dBth_dr *sqrtBR2BT2 - Bth * dsqrtBR2BT2_dr )/(sqrtBR2BT2*sqrtBR2BT2);
}

__device__ float eval_DsinZeta_dtheta(float SignAsun, float Bth, float dBth_dth, float sqrtBR2BT2, float dsqrtBR2BT2_dth) {

  return SignAsun*( dBth_dth*sqrtBR2BT2 - Bth * dsqrtBR2BT2_dth)/(sqrtBR2BT2*sqrtBR2BT2);
}

__device__ float eval_DcosZeta_dr(float r, float SignAsun, float PolSign, float sqrtBR2BT2, float dsqrtBR2BT2_dr) {

  return SignAsun*( (-2.*PolSign/(r)) *sqrtBR2BT2  - PolSign  * dsqrtBR2BT2_dr )/(sqrtBR2BT2*sqrtBR2BT2);
}

__device__ float eval_DcosZeta_dtheta(float SignAsun, float PolSign, float sqrtBR2BT2, float dsqrtBR2BT2_dth, float DelDirac) {

  return SignAsun*( (-2.*DelDirac)*sqrtBR2BT2  - PolSign * dsqrtBR2BT2_dth)/(sqrtBR2BT2*sqrtBR2BT2);
}


////////////////////////////////////////////////////////////////
//..... Simmetric component and derivative  ....................
////////////////////////////////////////////////////////////////

__device__ struct DiffusionTensor_t DiffusionTensor_symmetric(unsigned char InitZone, signed char HZone, float r, float th, float phi, float R,
                                    struct PartDescription_t pt, float GaussRndNumber){
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
    // ....... Get Diffusion tensor in HMF frame
    // Kpar = Kh.x    dKpar_dr = dK_dr.x  
    // Kp1  = Kh.y    dKp1_dr  = dK_dr.y
    // Kp2  = Kh.z    dKp2_dr  = dK_dr.z
    float3 dK_dr;
    float3 Kh = Diffusion_Tensor_In_HMF_Frame(InitZone,HZone,r,th,beta_R(R,pt),R,GaussRndNumber,dK_dr);
    // ....... Define the HMF Model
    // The implemented model is the Parker’s IMF with Jokipii and Kota (1989) modification in polar regions
    // if (fmod(fabs(th)-M_PI, M_PI)<1e-6) th = 1e-6;   // Correction for divergent Bth in polar regions

    float PolSign = (th-Pi/2.>0)? -1.:+1.;
    float SignAsun = (LIM[HZone+InitZone].Asun>0)?+1.:-1.;

    float V_SW  = SolarWindSpeed(InitZone, HZone, r, th, phi);
    
    float Bth = eval_Bth(r, th, IsPolarRegion);
    float Bph = eval_Bph(r, th, PolSign, V_SW);
    float HMF_Mag = eval_HMF_Mag(r, th, IsPolarRegion, Bth, Bph);

    float sinZeta = eval_sinZeta(PolSign, SignAsun, Bth);
    float cosZeta = eval_cosZeta(PolSign, SignAsun, Bth);
    float sinPsi = eval_sinPsi(SignAsun, Bph, HMF_Mag);
    float cosPsi = eval_cosPsi(PolSign, Bth, HMF_Mag);

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

    // Derivatives
    float sqrtBR2BT2 = eval_sqrtBR2BT2(PolSign, Bth);    
    float dBth_dr = eval_dBth_dr(IsPolarRegion, th);
    float dsqrtBR2BT2_dr = eval_dsqrtBR2BT2_dr(r, PolSign, Bth, sqrtBR2BT2, dBth_dr);
    float DelDirac= (th==Pi/2.)?1:0;
    float dBth_dth = eval_dBth_dth(IsPolarRegion, r, th);
    
    float dsqrtBR2BT2_dth = eval_dsqrtBR2BT2_dth(PolSign, Bth, sqrtBR2BT2, dBth_dth, DelDirac);
    float dBph_dr = eval_dBph_dr(r, th, PolSign, V_SW);
    float dBMag_dr = eval_dBMag_dr(r, PolSign, Bth, Bph, dBth_dr, dBph_dr, HMF_Mag);

    float DcosZeta_dr = eval_DcosZeta_dr (r, SignAsun, PolSign, sqrtBR2BT2, dsqrtBR2BT2_dr);
    float DcosZeta_dtheta = eval_DcosZeta_dtheta (SignAsun, PolSign, sqrtBR2BT2, dsqrtBR2BT2_dth, DelDirac);
    float DcosPsi_dr = eval_DcosPsi_dr (r, PolSign, Bth, Bph, sqrtBR2BT2, dBth_dr, dBph_dr, HMF_Mag);
    float DsinPsi_dr = eval_DsinPsi_dr(SignAsun, HMF_Mag, Bph, dBph_dr, dBMag_dr);

    // Here we apply some semplification due to HMF and Kdiff description
    // B field do not depend on phi
    // Kpar,Kperp1-2 do not depends on theta and phi
    if (IsPolarRegion){
      float dV_SWdth = DerivativeOfSolarWindSpeed_dtheta(InitZone, HZone, r, th, phi);
      float dBph_dth = eval_dBph_dth(r, th, PolSign, V_SW, dV_SWdth, DelDirac);
      float dBMag_dth = eval_dBMag_dth(PolSign, dV_SWdth, DelDirac, Bth, dBth_dth, Bph, dBph_dth, HMF_Mag);

      float DsinZeta_dr = eval_DsinZeta_dr (SignAsun, Bth, dBth_dr, sqrtBR2BT2, dsqrtBR2BT2_dr);
      float DsinZeta_dtheta = eval_DsinZeta_dtheta (SignAsun, Bth, dBth_dth, sqrtBR2BT2, dsqrtBR2BT2_dth);
      float DcosPsi_dtheta = eval_DcosPsi_dtheta (PolSign, Bth, Bph, sqrtBR2BT2, dBph_dth, dBth_dth, HMF_Mag, DelDirac);
      float DsinPsi_dtheta = eval_DsinPsi_dtheta(SignAsun, HMF_Mag, Bph, dBph_dth, dBMag_dth);

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
    }
    
    else{
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
 
  }
  
  else{
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
__device__ Tensor3D_t SquareRoot_DiffusionTerm(signed char HZone, float KSym_rr, float KSym_tr, float KSym_tt, float KSym_pr, float KSym_pt,
                      float KSym_pp, float r, float th){
/* * description:  ========== Decomposition of Diffusion Tensor ====== 
                  solve square root of diffusion tensor in heliocentric spherical coordinates
    \par Tensor3D K     input tensor
    \par qvect_t  part  particle Position
    \par int      &res  0 if ok; 1 if error
   */

    // // ---- Create diffusion matrix of FPE from diffusion tensor
    float sintheta=sin(th);
    if (HZone<Heliosphere.Nregions) {
                                        //KSym_rt=2.*KSym_rt/r;                       KSym_rp=2.*KSym_rp/(r*sintheta);
        KSym_tr=2.*KSym_tr/r;              KSym_tt=2.*KSym_tt/(r*r);           //KSym_tp=2.*KSym_tp/(r2*sintheta);
        KSym_pr=2.*KSym_pr/(r*sintheta);   KSym_pt=2.*KSym_pt/(r*r*sintheta);  KSym_pp=2.*KSym_pp/(r*r*sintheta*sintheta);     
    }

    // ---- square root of diffusion tensor in heliocentric spherical coordinates
    struct Tensor3D_t D;
    // // first try
    // // h=0 i=0 n=0
    // // *Ddif_rt = 0;
    // // *Ddif_rp = 0;
    // // *Ddif_tp = 0;
    D.rr = sqrt(2.*KSym_rr);                    // g = sqrt(a)
    if (HZone<Heliosphere.Nregions) {      // do this only in the inner heliosphere (questo perchè *Ddif_pp vale nan nell'outer heliosphere, forse legato alla radice di un elemento negativo (i.e. arrotondamenti allo zero non ottimali))
        D.tr = KSym_tr/ D.rr;                     // l = b/g
        D.pr = KSym_pr/ D.rr;                     // o = c/g
        D.tt = sqrt(KSym_tt - D.tr*D.tr);        // m = sqrt(d-l^2)
        D.pt = 1./ D.tt * ( KSym_pt - D.tr*D.pr ); // p = 1/m (e-lo)
        D.pp = sqrt( KSym_pp - D.pr*D.pr - D.pt*D.pt); // q = sqrt(f - o^2 -p^2)
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
__device__ float AdvectiveTerm_radius(float v_drift_rad, unsigned char InitZone, signed char HZone, float K_rr, float K_tr, float DKrr_dr, float DKtr_dt, float r, float th, float phi, float R, struct PartDescription_t pt){
  float AdvTerm;
  
  if (HZone<Heliosphere.Nregions) {
    // inner Heliosphere .........................
    float tantheta=tan(th);
    // advective part related to diffision tensor
    AdvTerm   = 2.* K_rr/r + DKrr_dr + K_tr/(r*tantheta) + DKtr_dt/r ;

    // drift component
    AdvTerm   += - v_drift_rad  ;
  }
  
  else {
    // heliosheat ...............................
    // advective part related to diffision tensor
    AdvTerm   = 2.* K_rr/r + DKrr_dr ;
    // drift component     
    // -- none --
  }

  // convective part related to solar wind
  AdvTerm  += -SolarWindSpeed(InitZone, HZone, r, th, phi);

  return AdvTerm;
}

__device__ float AdvectiveTerm_theta(float v_drift_th, unsigned char InitZone,signed char HZone, float K_tr, float K_tt, float DKrt_dr, float DKtt_dt, float r, float th, float phi, float R, struct PartDescription_t pt){
  float AdvTerm;
  if (HZone<Heliosphere.Nregions) {
    // inner Heliosphere .........................
    float tantheta=tan(th);
    float r2      = r*r;
    // advective part related to diffision tensor
    AdvTerm  = K_tr/(r2) + K_tt/(tantheta*r2 ) + DKrt_dr/r + DKtt_dt/r2 ;       

    // drift component
    AdvTerm  += - v_drift_th/r ;
  }
  
  else {
    // heliosheat ...............................
    // advective part related to diffision tensor
    AdvTerm  = 0;       
    // drift component     
    // -- none --
  }

  return AdvTerm;
}

__device__ float AdvectiveTerm_phi(float v_drift_phi, unsigned char InitZone,signed char HZone, float K_pr, float DKrp_dr, float DKtp_dt, float r, float th, float phi, float R, struct PartDescription_t pt){
  float AdvTerm;
  if (HZone<Heliosphere.Nregions) {
    // inner Heliosphere .........................
    float sintheta=sin(th);
    float tantheta=tan(th);
    float r2      = r*r;
    // advective part related to diffision tensor

    AdvTerm = K_pr/(r2*sintheta)+ DKrp_dr/(r*sintheta) + DKtp_dt/( r2*sintheta)  ;
    // drift component
    AdvTerm += - v_drift_phi/(r*sintheta);
  }
  
  else {
    // heliosheat ...............................
    // advective part related to diffision tensor
    AdvTerm = 0;
    // drift component     
    // -- none --
  }

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