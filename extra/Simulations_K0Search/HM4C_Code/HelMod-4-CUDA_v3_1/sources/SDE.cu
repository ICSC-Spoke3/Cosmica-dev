#ifndef GLOBALS
#include "globals.h"
#endif
#include "SolarWind.h"
#include "HeliosphereModel.h"
#include "MagneticDrift.h"
#include "SDE.h"


#include <stdio.h>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions

  ////////////////////////////////////////////////////////////////
  //..... Energy loss term   .....................................
  ////////////////////////////////////////////////////////////////
__device__ float EnergyLoss(unsigned short InitZone, signed short HZone, particle_t ev)
{ 
  if (HZone<Heliosphere.Nregions){
   // inner Heliosphere .........................
   return 2./3.*SolarWindSpeed(InitZone,HZone,ev.part)/ev.part.r * (ev.part.Ek + 2.*ev.pt.T0)/(ev.part.Ek + ev.pt.T0) * ev.part.Ek;
  }else{
   // heliosheat ...............................
   // no energy loss
   return 0;
  }
}


  ////////////////////////////////////////////////////////////////
  //..... Loss term  .............................................
  ////////////////////////////////////////////////////////////////
__device__ float LossTerm(unsigned short InitZone,signed short HZone, particle_t ev ){
   // xxx da completare
  if (HZone<Heliosphere.Nregions){
    // inner Heliosphere .........................
    return 2.*SolarWindSpeed(InitZone,HZone,ev.part)/ev.part.r *( 1./3.*( ev.part.Ek*ev.part.Ek+2*ev.part.Ek*ev.pt.T0+ 2*ev.pt.T0*ev.pt.T0 )/( (ev.part.Ek+ev.pt.T0)*(ev.part.Ek+ev.pt.T0) ) -1 );
   }else{
    // heliosheat ...............................
    // solar wind divergence is zero
    return 0;
   }
  return 1;   
}

  ////////////////////////////////////////////////////////////////
  //.....  Diffusion Term  .....................................
  ////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// ========== Decomposition of Diffution Tensor ======
// solve the equation (square root of diffusion tensor)
//  | a b c |   | g h i | | g l o |
//  | b d e | = | l m n | | h m p |
//  | c e f |   | o p q | | i n q |
// 
// the diffusion tensor is written in the form that arise from SDE calculation
///////////////////////////////////////////////////////
__device__ Tensor3D_t SquareRoot_DiffusionTerm(signed short HZone,Tensor3D_t K, qvect_t part, int &res ){
/* * description:  ========== Decomposition of Diffusion Tensor ====== 
                  solve square root of diffusion tensor in heliocentric spherical coordinates
    \par Tensor3D K     input tensor
    \par qvect_t  part  particle Position
    \par int      &res  0 if ok; 1 if error
   */

   // // ---- Create diffusion matrix of FPE from diffusion tensor
   float sintheta=sin(part.th);
   K.rr=2.*K.rr;                    
   if (HZone<Heliosphere.Nregions) {
                                        //K.rt=2.*K.rt/r;                       K.rp=2.*K.rp/(r*sintheta);
      K.tr=2.*K.tr/part.r;              K.tt=2.*K.tt/(part.r*part.r);           //K.tp=2.*K.tp/(r2*sintheta);
      K.pr=2.*K.pr/(part.r*sintheta);   K.pt=2.*K.pt/(part.r*part.r*sintheta);  K.pp=2.*K.pp/(part.r*part.r*sintheta*sintheta);     
   }
   // ---- square root of diffusion tensor in heliocentric spherical coordinates
   Tensor3D_t D;
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
    ||(isnan(D.rr))||(isnan(D.tr))||(isnan(D.pr))||(isnan(D.tt))||(isnan(D.pt))||(isnan(D.pp)) )
   { // there was some error...
    // -- TODO -- check an other solution... see Pei et al 2010 or Kopp et al 2012
    // not implemented since such cases are rare
      res =1;
   }
   else
   {
      res = 0;
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
__device__ vect3D_t AdvectiveTerm(unsigned short InitZone,signed short HZone, DiffusionTensor_t K, qvect_t part ,PartDescription_t pt){
   vect3D_t AdvTerm;
   if (HZone<Heliosphere.Nregions){
      // inner Heliosphere .........................
      float sintheta=sin(part.th);
      float tantheta=tan(part.th);
      float r2      = part.r*part.r;
      // advective part related to diffision tensor
      AdvTerm.r   = 2.* K.K.rr/part.r + K.DKrr_dr + K.K.tr/(part.r*tantheta) + K.DKtr_dt/part.r ;
      AdvTerm.th  = K.K.tr/(r2) + K.K.tt/(tantheta*r2 ) + K.DKrt_dr/part.r + K.DKtt_dt/r2 ;       

      AdvTerm.phi = K.K.pr/(r2*sintheta)+ K.DKrp_dr/(part.r*sintheta) + K.DKtp_dt/( r2*sintheta)  ;
      // drift component
      vect3D_t v_drift = Drift_PM89(InitZone, HZone,  part, pt);
      AdvTerm.r   += - v_drift.r  ;
      AdvTerm.th  += - v_drift.th/part.r ;
      AdvTerm.phi += - v_drift.phi/(part.r*sintheta);
   
   }else{
      // heliosheat ...............................
      // advective part related to diffision tensor
      AdvTerm.r   = 2.* K.K.rr/part.r + K.DKrr_dr ;
      AdvTerm.th  = 0;       
      AdvTerm.phi = 0;
      // drift component     
      // -- none --
   }
   // convective part related to solar wind
   AdvTerm.r  += -SolarWindSpeed(InitZone,HZone,part);

   return AdvTerm;
}