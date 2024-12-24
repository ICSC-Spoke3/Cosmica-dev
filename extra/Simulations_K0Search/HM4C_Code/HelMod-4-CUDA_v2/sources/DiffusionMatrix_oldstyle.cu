#ifndef GLOBALS
#include "globals.h"
#endif
#include "SolarWind.h"
#include "DiffusionModel.h"
#include "DiffusionMatrix.h"



  ////////////////////////////////////////////////////////////////
  //..... Simmetric component and derivative  ....................
  ////////////////////////////////////////////////////////////////

__device__ DiffusionTensor_t DiffusionTensor_symmetric(unsigned short InitZone,signed short HZone, qvect_t part,PartDescription_t pt,float GaussRndNumber ){
  /*Authors: 2022 Stefano */ 
  /* * description: Evaluate the symmetric component (and derivative) of diffusion tensor in heliocentric coordinates
      \param InitZone initial Zone in the heliosphere (in the list of parameters)
      \param HZone   Zone in the Heliosphere
      \param part    particle position and energy
      \return  the symmetric component (and derivative) of diffusion tensor
   */
  DiffusionTensor_t KK;
  if (HZone<Heliosphere.Nregions){
    /*  NOTE about HMF
     *  In the equatorial region, we used the Parker’s IMF (B Par ) in the parametrization of Hattingh and Burger (1995), 
     *  while in the polar regions we used a modiﬁed IMF (B Pol ) that includes a latitudinal component, 
     *  accounting for large scale ﬂuctuations, dominant at high heliolatitudes, as suggested by Jokipii and Kota (1989).
     *  The polar region is defined by the constant 'PolarZone' and 'CosPolarZone' (globals.h)
     *  Note that since in equatorial region the theta component of HMF is zero we implemented two cases to avoid 0calculations.
     */
    bool IsPolarRegion = (fabs(cos(part.th))>CosPolarZone )? true:false;
    float SignAsun = (LIM[HZone+InitZone].Asun>0)?+1.:-1.;
    // ....... Get Diffusion tensor in HMF frame
    // Kpar = Kh.x    dKpar_dr = dK_dr.x  
    // Kp1  = Kh.y    dKp1_dr  = dK_dr.y
    // Kp2  = Kh.z    dKp2_dr  = dK_dr.z
    float3 dK_dr;
    float3 Kh = Diffusion_Tensor_In_HMF_Frame(InitZone,HZone,part.r,part.th,beta_(part.Ek,pt.T0),Rigidity(part.Ek,pt),GaussRndNumber,dK_dr);

    
    // ....... Define the HMF Model
    // The implemented model is the Parker’s IMF with Jokipii and Kota (1989) modification in polar regions
    int PolSign = (part.th-Pi/2.>0)? -1.:+1.;
    int DelDirac= (part.th==Pi/2.)?1:0;
    float V_SW  = SolarWindSpeed(InitZone,HZone, part );

    float Br = float(PolSign);                                                // divided by A/r^2
    float Bth= (IsPolarRegion)?part.r*delta_m/(rhelio*sin(part.th)):0.;       // divided by A/r^2
    float Bph= - float(PolSign)*((Omega*(part.r-rhelio)*sin(part.th))/V_SW);  // divided by A/r^2

    float dBr_dr   = -2.*PolSign/(part.r);
    float dBr_dth  = -2.*float(DelDirac);

    float dBth_dr  = (IsPolarRegion)?-delta_m/(rhelio*sin(part.th)):0.;
    float dBth_dth = (IsPolarRegion)?part.r*delta_m/(rhelio*sin(part.th)*sin(part.th))*(-cos(part.th)):0.;

    float dBph_dr  = float(PolSign) *(part.r-2.*rhelio)*(Omega*sin(part.th))/(part.r*V_SW);
    float dBph_dth = - (part.r-rhelio)*Omega* ( -PolSign*( cos(part.th) *V_SW - sin(part.th)*DerivativeOfSolarWindSpeed_dtheta(InitZone,HZone, part) ) + 2. * sin(part.th) *V_SW*float(DelDirac)) /(V_SW*V_SW);
         
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

    float sinPsi2  = sinPsi  *sinPsi  ;
    float cosPsi2  = cosPsi  *cosPsi  ;
    float sinZeta2 = sinZeta *sinZeta ;
    float cosZeta2 = cosZeta *cosZeta ;
    
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
     * KK.rr = Kp2*sinZeta2 + cosZeta2 * (Kpar*cosPsi2 + Kp3*sinPsi2);
     * KK.tt = Kp2*cosZeta2 + sinZeta2 * (Kpar*cosPsi2 + Kp3*sinPsi2);
     * KK.pp = Kpar*sinPsi2 + Kp3*cosPsi2;
     * KK.rt = sinZeta*cosZeta*(Kpar*cosPsi2+Kp3*sinPsi2-Kp2);
     * KK.tr = sinZeta*cosZeta*(Kpar*cosPsi2+Kp3*sinPsi2-Kp2);
     * KK.rp = -(Kpar-Kp3) *sinPsi*cosPsi*cosZeta;
     * KK.pr = -(Kpar-Kp3) *sinPsi*cosPsi*cosZeta;
     * KK.tp = -(Kpar-Kp3) *sinPsi*cosPsi*sinZeta;
     * KK.pt = -(Kpar-Kp3) *sinPsi*cosPsi*sinZeta;
     */
    
    if (IsPolarRegion){
      // polar region
      KK.K.rr = Kh.y*sinZeta2 + cosZeta2 * (Kh.x*cosPsi2 + Kh.z*sinPsi2);
      KK.K.tt = Kh.y*cosZeta2 + sinZeta2 * (Kh.x*cosPsi2 + Kh.z*sinPsi2);
      KK.K.pp = Kh.x*sinPsi2 + Kh.z*cosPsi2;
      KK.K.tr = sinZeta*cosZeta*(Kh.x*cosPsi2+Kh.z*sinPsi2-Kh.y);
      KK.K.pr = -(Kh.x-Kh.z) *sinPsi*cosPsi*cosZeta;
      KK.K.pt = -(Kh.x-Kh.z) *sinPsi*cosPsi*sinZeta;
    }else{
      // equatorial region. Bth = 0 
      // --> sinZeta = 0 
      // --> cosZeta = +/- 1
      KK.K.rr =               + cosZeta2 * (Kh.x*cosPsi2 + Kh.z*sinPsi2);
      KK.K.tt = Kh.y*cosZeta2 ;
      KK.K.pp = Kh.x*sinPsi2 + Kh.z*cosPsi2;
      KK.K.tr = 0.;
      KK.K.pr = -(Kh.x-Kh.z) *sinPsi*cosPsi*cosZeta;
      KK.K.pt = 0.;
    }

    // ....... evaluate derivative of diffusion tensor
    /*  The complete calculations of derivatives are the follow:
     * note: in this case the diff tens is [Kpar,Kper2,Kper3]
     *
     *  KK.DKrr_dr = 2. * cosZeta*(cosPsi2*Kpar+Kp3*sinPsi2)*DcosZeta_dr+sinZeta2*DKp2_dr+cosZeta2*(2. * cosPsi*Kpar*DcosPsi_dr+cosPsi2*DKpar_dr+sinPsi* (sinPsi*DKp3_dr+2.*Kp3*DsinPsi_dr )) + 2.*Kp2*sinZeta*DsinZeta_dr;
     *  KK.DKtt_dt = 2. * cosZeta*Kp2*DcosZeta_dtheta+cosZeta2 * DKp2_dtheta + sinZeta2 * (2.*cosPsi*Kpar*DcosPsi_dtheta+cosPsi2*DKpar_dtheta+sinPsi*(sinPsi*DKp3_dtheta+2.*Kp3*DsinPsi_dtheta))+2.*(cosPsi2*Kpar+Kp3*sinPsi2)*sinZeta*DsinZeta_dtheta;
     *  KK.DKpp_dp = 2.*cosPsi*Kp3*DcosPsi_dphi + cosPsi2 * DKp3_dphi + sinPsi*(sinPsi*DKpar_dphi+2. *Kpar * DsinPsi_dphi);
     *  KK.DKrt_dr = (-Kp2+cosPsi2*Kpar+Kp3*sinPsi2)*sinZeta*DcosZeta_dr    +cosZeta*sinZeta*(2.*cosPsi*Kpar*DcosPsi_dr     - DKp2_dr    +cosPsi2*DKpar_dr    +sinPsi*(sinPsi*DKp3_dr    +2.*Kp3*DsinPsi_dr    ))+cosZeta*(-Kp2+cosPsi2*Kpar+Kp3*sinPsi2)*DsinZeta_dr    ;
     *  KK.DKtr_dt = (-Kp2+cosPsi2*Kpar+Kp3*sinPsi2)*sinZeta*DcosZeta_dtheta+cosZeta*sinZeta*(2.*cosPsi*Kpar*DcosPsi_dtheta - DKp2_dtheta+cosPsi2*DKpar_dtheta+sinPsi*(sinPsi*DKp3_dtheta+2.*Kp3*DsinPsi_dtheta))+cosZeta*(-Kp2+cosPsi2*Kpar+Kp3*sinPsi2)*DsinZeta_dtheta;
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
      KK.DKrr_dr = 2. * cosZeta*(cosPsi2*Kh.x+Kh.z*sinPsi2)*DcosZeta_dr + sinZeta2*dK_dr.y + cosZeta2*(2. * cosPsi*Kh.x*DcosPsi_dr+cosPsi2*dK_dr.x+ sinPsi* (sinPsi*dK_dr.z+2.*Kh.z*DsinPsi_dr )) + 2.*Kh.y*sinZeta*DsinZeta_dr;
      KK.DKtt_dt = 2. * cosZeta*Kh.y*DcosZeta_dtheta + sinZeta2 * (2.*cosPsi*Kh.x*DcosPsi_dtheta+2.*sinPsi*Kh.z*DsinPsi_dtheta)+2.*(cosPsi2*Kh.x+Kh.z*sinPsi2)*sinZeta*DsinZeta_dtheta;
      // KK.DKpp_dp = 0. ;
      KK.DKrt_dr = (-Kh.y+cosPsi2*Kh.x+Kh.z*sinPsi2)*(sinZeta*DcosZeta_dr    +cosZeta*DsinZeta_dr     )+cosZeta*sinZeta*(2.*cosPsi*Kh.x*DcosPsi_dr     +cosPsi2*dK_dr.x - dK_dr.y +sinPsi*(sinPsi*dK_dr.z    +2.*Kh.z*DsinPsi_dr    ));
      KK.DKtr_dt = (-Kh.y+cosPsi2*Kh.x+Kh.z*sinPsi2)*(sinZeta*DcosZeta_dtheta+cosZeta*DsinZeta_dtheta )+cosZeta*sinZeta*(2.*cosPsi*Kh.x*DcosPsi_dtheta                             +2.*sinPsi*Kh.z*DsinPsi_dtheta)                  ;
      KK.DKrp_dr = cosZeta * (Kh.z-Kh.x) * sinPsi * DcosPsi_dr    + cosPsi  * (Kh.z-Kh.x) *sinPsi * DcosZeta_dr + cosPsi* cosZeta* sinPsi* (dK_dr.z-dK_dr.x) + cosPsi * cosZeta*(Kh.z-Kh.x) * DsinPsi_dr;//-----------qui
      // KK.DKpr_dp = 0. ;
      KK.DKtp_dt = (Kh.z - Kh.x) * (sinPsi*sinZeta*DcosPsi_dtheta + cosPsi*(sinZeta*DsinPsi_dtheta + sinPsi*DsinZeta_dtheta) );
      // KK.DKpt_dp = 0. ;
    }else{
      // equatorial region. Bth = 0 
      // --> sinZeta = 0 
      // --> cosZeta = 1
      // --> derivative of sinZeta or cosZeta = 0
      KK.DKrr_dr = 2. * cosZeta*(cosPsi2*Kh.x+Kh.z*sinPsi2)*DcosZeta_dr +  cosZeta2*(2. * cosPsi*Kh.x*DcosPsi_dr+cosPsi2*dK_dr.x+ sinPsi* (sinPsi*dK_dr.z+2.*Kh.z*DsinPsi_dr )) ;
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
    KK.K.rr = Diffusion_Coeff_heliosheat(InitZone,part,beta_(part.Ek,pt.T0),Rigidity(part.Ek,pt),KK.DKrr_dr);
  }
  return KK;
}

