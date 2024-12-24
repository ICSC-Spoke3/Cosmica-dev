#ifndef GLOBALS
#include "globals.h"
#endif
#include "HeliosphereModel.h"
#include <stdio.h>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions


/////////////////////////////////////////////////////////////////////////////
////////////////// Diffusion Parameters Description /////////////////////////
/////////////////////////////////////////////////////////////////////////////
void RescaleToEffectiveHeliosphere(HeliosphereBoundRadius_t &Rbound,vect3D_t &part)
{
  /* * description: create an effective heliosphere of 100 AU. thisis due to the fact that K0 parameters are tuned on such dimension.
       \param  Rbound heliospher boundaries to be rescaled
       \param  part   initial position to be rescaled
   */
  float Rts_nose_realworld=Rbound.Rts_nose;
  float Rhp_nose_realworld=Rbound.Rhp_nose;
  float Rts_tail_realworld=Rbound.Rts_tail;
  float Rhp_tail_realworld=Rbound.Rhp_tail;
  
  Rbound.Rts_nose = 100.;
  Rbound.Rts_tail = Rts_tail_realworld*Rbound.Rts_nose/Rts_nose_realworld;

  Rbound.Rhp_nose = Rbound.Rts_nose+(Rhp_nose_realworld-Rts_nose_realworld); //122.;
  Rbound.Rhp_tail = Rbound.Rts_tail+(Rhp_tail_realworld-Rts_tail_realworld); //Rhp_tail*Rhp/Rhp_realworld;

  float HM_Rts_d = Boundary(part.th,part.phi, Rbound.Rts_nose, Rbound.Rts_tail);
  float RW_Rts_d = Boundary(part.th,part.phi, Rts_nose_realworld, Rts_tail_realworld);
  float Rdi_real = part.r;
  if (Rdi_real<=RW_Rts_d) part.r=Rdi_real/RW_Rts_d*HM_Rts_d;
  else                    part.r=HM_Rts_d+ (Rdi_real-RW_Rts_d);
}


float K0Fit_ssn(int p, int SolarPhase, float ssn, float *GaussVar){
  /*Authors: 2011 Stefano - update 2012 Stefano - update 2015 GLV*/
  /* * description: K0 evaluated using ssn as a proxy
       \param p            solar polarity of HMF
       \param SolarPhase   0=rising / 1=Declining phase of solar activity cycle
       \param ssn          smoothed sunspot number
       \param *GaussVar    gaussian variation (output)
   */
  float k0;
  if(p>0.){
    if(SolarPhase==0)/*Rising*/   {k0=0.0002262-5.058e-7*ssn;   *GaussVar=0.1153;}
    else             /*Declining*/{k0=0.0002267-7.118e-7*ssn;   *GaussVar=0.1607;}
  }else{
    if(SolarPhase==0)/*Rising*/   {k0=0.0003059-2.51e-6*ssn+1.284e-8*ssn*ssn-2.838e-11*ssn*ssn*ssn;   *GaussVar=0.1097;}
    else             /*Declining*/{k0=0.0002876-3.715e-6*ssn+2.534e-8*ssn*ssn-5.689e-11*ssn*ssn*ssn;   *GaussVar=0.14;}
  }
  return k0;
}

float K0Fit_NMC(float NMC, float *GaussVar){
  /*Authors: 2015 GLV*/
  /* * description: K0 evaluated using Mc Murdo NM counts as a proxy
                    only for High Activity, defined as Tilt L >48deg
     \param NMC          Neutron monitor counting rate from Mc Murdo 
     \param *GaussVar    gaussian variation (output)
  */
  *GaussVar=0.1045;
  return exp(-10.83 -0.0041*NMC +4.52e-5*NMC*NMC);
}

float K0CorrFactor(int p, int q, int SolarPhase, float tilt){
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
  #define K0Corr_maxv 3.
#endif
#ifndef K0Corr_minv
  #define K0Corr_minv 1.
#endif
#ifndef K0Corr_p0_asc
  #define K0Corr_p0_asc 18.
#endif
#ifndef K0Corr_p1_asc
  #define K0Corr_p1_asc 40.
#endif
#ifndef K0Corr_p0_des
  #define K0Corr_p0_des 5.
#endif
#ifndef K0Corr_p1_des
  #define K0Corr_p1_des 53.
#endif
#ifndef K0Corr_maxv_neg
  #define K0Corr_maxv_neg 0.7
#endif
#ifndef K0Corr_p0_asc_neg
  #define K0Corr_p0_asc_neg 5.8
#endif
#ifndef K0Corr_p1_asc_neg
  #define K0Corr_p1_asc_neg 47.
#endif
#ifndef K0Corr_p0_des_neg
  #define K0Corr_p0_des_neg 5.8
#endif
#ifndef K0Corr_p1_des_neg
  #define K0Corr_p1_des_neg 58.
#endif 

  if (q>0){
    if (q*p>0){
      if (SolarPhase==0){ 
        //ascending
        return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_asc, K0Corr_p0_asc, tilt);
      }else{ 
        //descending
        return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_des, K0Corr_p0_des, tilt); 
      }
    }else{
      return 1;
    }
  }
  if (q<0){
    if (q*p>0){
      if (SolarPhase==0){ 
        //ascending
        return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_asc, K0Corr_p0_asc, tilt);
      }else{ 
        //descending
        return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_des, K0Corr_p0_des, tilt);
      }
    }else{
      if (SolarPhase==0){ 
        //ascending
        return SmoothTransition(K0Corr_maxv_neg, K0Corr_minv, K0Corr_p1_asc_neg, K0Corr_p0_asc_neg, tilt);
      }else{ 
        //descending
        return SmoothTransition(K0Corr_maxv_neg, K0Corr_minv, K0Corr_p1_des_neg, K0Corr_p0_des_neg, tilt);  
      }
    }                       
  }
  return 1;
}


float3 EvalK0(bool IsHighActivityPeriod, int p, int q, int SolarPhase, float tilt, float NMC,float ssn, unsigned char verbose=0){
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
  output.x=K0CorrFactor(p,q,SolarPhase,tilt); // k0_paral is corrected by a correction factor
////////////////////////////////////////////////
// printf("-- p: %d q: %d phase: %d tilt: %e ssn: %e NMC: %e \n",p,q,SolarPhase,tilt,ssn,NMC);  
// printf("-- K0CorrF: %e \n",output.x);
// printf("-- IsHighActivityPeriod %d \n",IsHighActivityPeriod);
////////////////////////////////////////////////     
  if (IsHighActivityPeriod && NMC>0){
    output.y=K0Fit_NMC(NMC, &output.z);
    output.x*=output.y;
  }else{
    if (verbose>=VERBOSE_med && IsHighActivityPeriod && NMC==0) { fprintf(stderr, "WARNING:: High Activity period require NMC variable setted with value >0, used ssn instead.\n");}
    output.y=K0Fit_ssn(p, SolarPhase, ssn, &output.z);
    output.x*=output.y;
  }
////////////////////////////////////////////////
// printf("-- K0 paral: %e \n",output.x);
// printf("-- K0 perp : %e \n",output.y);
////////////////////////////////////////////////       
  return output;
}

float g_low(int SolarPhase, float tilt){
  /*Authors: 2022 Stefano */
  /* * description: evaluate g_low parameter (for Kparallel).
      \param SolarPhase   0=rising / 1=Declining phase of solar activity cycle
      \param tilt         Tilt angle of neutral sheet (in degree)
      \return g_low
  */
#ifndef MaxValueOf_g_low
  #define MaxValueOf_g_low 0.5
#endif  
#ifndef CAB_TransPoint_des
  #define CAB_TransPoint_des 45
#endif 
#ifndef CAB_smoothness_des
  #define CAB_smoothness_des 10.
#endif 
#ifndef CAB_TransPoint_asc
  #define CAB_TransPoint_asc 60
#endif 
#ifndef CAB_smoothness_asc
  #define CAB_smoothness_asc 9.
#endif   
  float g_low = 0;
  if (SolarPhase==1){
    g_low=MaxValueOf_g_low*SmoothTransition(1, 0, CAB_TransPoint_des, CAB_smoothness_des, tilt);
  }else{
    g_low=MaxValueOf_g_low*SmoothTransition(1, 0, CAB_TransPoint_asc, CAB_smoothness_asc, tilt);
  }
  return g_low;
}

float rconst(int SolarPhase, float tilt){
  /*Authors: 2022 Stefano */
  /* * description: evaluate rconst parameter (for Kparallel).
      \param SolarPhase   0=rising / 1=Declining phase of solar activity cycle
      \param tilt         Tilt angle of neutral sheet (in degree)
      \return rconst
  */
#ifndef MaxValueOf_rconst
  #define MaxValueOf_rconst 4
#endif  
#ifndef rconst_TransPoint_des
  #define rconst_TransPoint_des 45
#endif 
#ifndef rconst_smoothness_des
  #define rconst_smoothness_des 10.
#endif 
#ifndef rconst_TransPoint_asc
  #define rconst_TransPoint_asc 60
#endif 
#ifndef rconst_smoothness_asc
  #define rconst_smoothness_asc 9.
#endif   
  float rconst = 0;
  if (SolarPhase==1){
    rconst=SmoothTransition(MaxValueOf_rconst, 1, rconst_TransPoint_des, rconst_smoothness_des, tilt);
  }else{
    rconst=SmoothTransition(MaxValueOf_rconst, 1, rconst_TransPoint_asc, rconst_smoothness_asc, tilt);
  }
  return rconst;
}

__device__ float3 Diffusion_Tensor_In_HMF_Frame(unsigned short InitZone,signed short HZone, float r, float theta, float beta, float P, float GaussRndNumber, float3 &dK_dr) {
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
  HeliosphereZoneProperties_t ThisZone=LIM[HZone+InitZone];

  // Kpar = k0 * beta/3 * (P/1GV + glow)*( Rconst+r/1AU) with k0 gaussian distributed
  Ktensor.x = (ThisZone.k0_paral[Heliosphere.IsHighActivityPeriod[InitZone]?0:1] + GaussRndNumber*ThisZone.GaussVar[Heliosphere.IsHighActivityPeriod[InitZone]?0:1]*ThisZone.k0_paral[Heliosphere.IsHighActivityPeriod[InitZone]?0:1] ) ;
  dK_dr.x   = Ktensor.x;
  Ktensor.x *= beta/3. * (P+ThisZone.g_low) * (ThisZone.rconst+r);
  dK_dr.x   *= beta/3. * (P+ThisZone.g_low) ;

#ifndef rho_1
  #define rho_1 0.065 // Kpar/Kperp (ex Kp0)
#endif
  // Kperp1 = rho_1(theta)* k0 * beta/3 * (P/1GV + glow)*( Rconst+r/1AU)
  Ktensor.y = rho_1 * ThisZone.k0_perp[Heliosphere.IsHighActivityPeriod[InitZone]?0:1] * beta/3. * (P+ThisZone.g_low) * (ThisZone.rconst+r);
  dK_dr.y   = rho_1 * ThisZone.k0_perp[Heliosphere.IsHighActivityPeriod[InitZone]?0:1] * beta/3. * (P+ThisZone.g_low) ;
#ifndef PolarEnhanc
  #define PolarEnhanc 2 // polar enhancement in polar region
#endif
  if (fabs(cos(theta))>CosPolarZone ) {
    Ktensor.y*=PolarEnhanc; // applied a sharp transition
    dK_dr.y  *=PolarEnhanc; // applied a sharp transition
  }

  // Kperp2 = rho_2 * k0 * beta/3 * (P/1GV + glow)*( Rconst+r/1AU) with rho_2=rho_1
  Ktensor.z = rho_1 * ThisZone.k0_perp[Heliosphere.IsHighActivityPeriod[InitZone]?0:1] * beta/3. * (P+ThisZone.g_low) * (ThisZone.rconst+r);
  dK_dr.z   = rho_1 * ThisZone.k0_perp[Heliosphere.IsHighActivityPeriod[InitZone]?0:1] * beta/3. * (P+ThisZone.g_low) ;
  return Ktensor;
}

__device__ float Diffusion_Coeff_heliosheat(signed short HZone, qvect_t part, float beta, float P, float &dK_dr) {
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
  float RhpDirection=Boundary(part.th, part.phi,Heliosphere.RadBoundary_effe[HZone].Rhp_nose, Heliosphere.RadBoundary_effe[HZone].Rhp_tail);
#ifndef HPB_SupK
  #define HPB_SupK 50 // suppressive factor at barrier
#endif  
#ifndef HP_width
  #define HP_width 2 // amplitude in AU of suppressive factor at barrier
#endif  
#ifndef HP_SupSmooth
  #define HP_SupSmooth 3e-2 // smoothness of suppressive factor at barrier
#endif  
  if ((part.r>RhpDirection-5))
                        {
      return HS[HZone].k0*beta*(P)*SmoothTransition(1, 1./HPB_SupK, RhpDirection-(HP_width/2.), HP_SupSmooth,part.r);
                        }
  else {
      return HS[HZone].k0*beta*(P);
  }
}