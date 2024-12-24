#ifndef GLOBALS
#include "globals.h"
#endif

#include "HeliosphereModel.h"
#include "SolarWind.h"

  ////////////////////////////////////////////////////////////////
  //..... Solar Wind Speed   .....................................
  ////////////////////////////////////////////////////////////////
__device__ float SolarWindSpeed(unsigned short InitZone,signed short HZone, qvect_t part){
   /* Author: SDT - adapted for CUDA in Feb 2022
   * description: return the solar wind speed
   *              
   * \param HZone Index of Heliosphere region
   * \param part  Pseudoparticle position
   */
   float V0=(HZone<Heliosphere.Nregions)?LIM[HZone+InitZone].V0:HS[InitZone].V0;
   // heliosheat (or near to)...............................
   if (HZone>=Heliosphere.Nregions-1){
      float RtsDirection  = Boundary(part.th,part.phi,Heliosphere.RadBoundary_effe[InitZone].Rts_nose, Heliosphere.RadBoundary_effe[InitZone].Rts_tail);
      if (part.r>RtsDirection-L_tl){
         float RtsRWDirection= Boundary(part.th,part.phi,Heliosphere.RadBoundary_real[InitZone].Rts_nose, Heliosphere.RadBoundary_real[InitZone].Rts_tail);    
         float DecreasFactor = SmoothTransition(1., 1./s_tl, RtsDirection, L_tl, part.r);
         if (part.r>RtsDirection){
            DecreasFactor*=pow(RtsRWDirection/(RtsRWDirection -RtsDirection + part.r),2.); 
         }
         return V0*DecreasFactor;
      }
   }
   // inner Heliosphere .........................
   if (Heliosphere.IsHighActivityPeriod[InitZone]){
      // high solar activity
      return V0;
   }else{
      // low solar activity
      float VswAngl=1;
      if (Vhigh/V0<=2.){VswAngl=Vhigh/V0-1.;}
      if (fabs(cos(part.th))>VswAngl){
         return Vhigh ;
      }else{ 
         return (V0*(1+fabs(cos(part.th))));
      }
   }
   return V0;
}
__device__ float DerivativeOfSolarWindSpeed_dtheta(unsigned short InitZone,signed short HZone, qvect_t part){
   /* Author: SDT - adapted for CUDA in Feb 2022
   * description: return the derivative of solar wind speed in d theta
   *              
   * \param HZone Index of Heliosphere region
   * \param part  Pseudoparticle position
   */
   float V0= (HZone<Heliosphere.Nregions)?LIM[HZone+InitZone].V0:HS[InitZone].V0;
   // inner Heliosphere .........................
   if (Heliosphere.IsHighActivityPeriod[InitZone]){
      // high solar activity
      return 0;
   }else{
      // low solar activity
      float VswAngl=1;
      if (Vhigh/V0<=2.){VswAngl=Vhigh/V0-1.;}
      if (fabs(cos(part.th))>VswAngl){
         return 0 ;
      }else{ 
         if (cos(part.th)<0)  return V0*sin(part.th);
         if (cos(part.th)==0) return 0.;
         if (cos(part.th)>0)  return -V0*sin(part.th);
      }
   }
   // heliosheat ...............................
   if (HZone>=Heliosphere.Nregions-1){
      float RtsDirection  = Boundary(part.th,part.phi,Heliosphere.RadBoundary_effe[InitZone].Rts_nose, Heliosphere.RadBoundary_effe[InitZone].Rts_tail);
      if (part.r>RtsDirection-L_tl){
         return 0;
      }
   }
   return 0;
}