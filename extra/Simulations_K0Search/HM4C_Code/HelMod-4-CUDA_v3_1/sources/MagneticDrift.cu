#ifndef GLOBALS
#include "globals.h"
#endif
#include "MagneticDrift.h"
#include "SolarWind.h"


#include <stdio.h>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions


  ////////////////////////////////////////////////////////////////
  //..... Magnetic Drift Model ..................................  
  //.... Potgieter Mooral 1985 - See Burger&Hatttingh 1995
  ////////////////////////////////////////////////////////////////
__device__ float Gamma_Bfield(float r, float th, float Vsw)
{
  return (Omega*(r-rhelio)*sin(th))/Vsw;
}
__device__ float delta_Bfield(float r, float th)
{
  return r/rhelio*delta_m/sin(th);;
}

__device__ vect3D_t Drift_PM89(unsigned short InitZone, signed short HZone, qvect_t part,PartDescription_t pt){
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
  bool IsPolarRegion = (fabs(cos(part.th))>CosPolarZone )? true:false;
  float Ka   = (pt.A*sqrt(part.Ek*(part.Ek+2*pt.T0))*(GeV)*beta_(part.Ek,pt.T0))/(3*pt.Z);  /* sign(Z)*beta*P/3 constant part of antisymmetric diffusion coefficient */
  float Asun = LIM[HZone+InitZone].Asun;                                           /* Magnetic Field Amplitude constant / aum^2*/
  float dV_dth = DerivativeOfSolarWindSpeed_dtheta(InitZone,HZone, part );
  float TiltAngle = LIM[HZone+InitZone].TiltAngle;
  float P = Rigidity(part.Ek,pt);
  // .. Scaling factor of drift in 2D approximation.  to account Neutral sheet
  float fth=0;        /* scaling function of drift vel */
  float Dftheta_dtheta =0;
  float theta_mez;             /* scaling parameter */
  qvect_t TiltPos=part;
  TiltPos.th=Pi/2.-TiltAngle;
  float Vsw  = SolarWindSpeed(InitZone,HZone, TiltPos );
  float dthetans = fabs((GeV/(c*aum))*(2.*part.r*(pt.A*sqrt(part.Ek*(part.Ek+2*pt.T0))))/(fabs(pt.Z)*(Asun)*sqrt( 1+Gamma_Bfield(part.r,TiltPos.th,Vsw)*Gamma_Bfield(part.r,TiltPos.th,Vsw)+((IsPolarRegion)?delta_Bfield(part.r,TiltPos.th)*delta_Bfield(part.r,TiltPos.th):0))));  /*dTheta_ns = 2*R_larmor/r*/
  //                                                                                    B_mag_alfa    = Asun/r2*sqrt(1.+ Gamma_alfa*Gamma_alfa + delta_alfa*delta_alfa );
  // double B2_mag_alfa   = B_mag_alfa*B_mag_alfa;
  //       dthetans = fabs((GeV/(c*aum))*(2.*       (MassNumber*sqrt(T*(T+2*T0))         ))/(Z*r*sqrt(B2_mag_alfa)));  /*dTheta_ns = 2*R_larmor/r*/
  
  if ((TiltAngle+dthetans)>Pi/2.) theta_mez=Pi/2.-0.5*sin(Pi/2.);
     else                                    theta_mez=Pi/2.-0.5*sin(TiltAngle+dthetans);
  if (theta_mez<(Pi/.2)){
    float a_f      = acos(Pi/(2.*theta_mez)-1);
    fth         = 1./(a_f)*atan((1.-(2.*part.th/Pi))*tan( a_f ) );
    Dftheta_dtheta = - 2. * tan(a_f)/( a_f*Pi*(1.+(1-2.*part.th/Pi)*(1-2.*part.th/Pi)* tan(a_f)* tan(a_f) ));
    }
   else{/* if the smoothness parameter "theta_mez" is greater then Pi/2, then the neutral sheet is flat, only heaviside function is applied.*/
        if (part.th>Pi/2)      { fth= 1.;}
        else {if (part.th<Pi/2){ fth=-1.;}
        else {if (part.th==Pi/2) fth= 0.;}}
        }
                                                                                                        // if (debug){
                                                                                                        //    printf("Vsw(tilt) %e\tBMagAlpha=%e\tAsun=%e\tr2=%e\tGamma_alfa=%e\tdelta_alfa=%e\n\n", Vsw,(Asun/(part.r*part.r))*sqrt( 1+Gamma_Bfield(part.r,TiltPos.th,Vsw)*Gamma_Bfield(part.r,TiltPos.th,Vsw)+delta_Bfield(part.r,TiltPos.th)*delta_Bfield(part.r,TiltPos.th)),
                                                                                                        //             Asun,part.r*part.r, Gamma_Bfield(part.r,TiltPos.th,Vsw), ((IsPolarRegion)?delta_Bfield(part.r,TiltPos.th)*delta_Bfield(part.r,TiltPos.th):0));
                                                                                                        //    printf("KA %f\tdthetans=%e\tftheta=%e\tDftheta_dtheta=%e\n", Ka, dthetans,fth,Dftheta_dtheta);
                                                                                                        // }
  Vsw  = SolarWindSpeed(InitZone,HZone, part ); // solar wind evaluated 
  // .. drift velocity

  if (IsPolarRegion){
      // polar region
      float E = delta_m*delta_m*part.r*part.r*Vsw*Vsw+Omega*Omega*rhelio*rhelio*(part.r-rhelio)*(part.r-rhelio)*sin(part.th)*sin(part.th)*sin(part.th)*sin(part.th)+rhelio*rhelio*Vsw*Vsw*sin(part.th)*sin(part.th) ;
      float C = fth*sin(part.th)*Ka*part.r*rhelio/(Asun* E*E )  ;
      C *= P*P/(P*P+ LIM[HZone+InitZone].P0d*LIM[HZone+InitZone].P0d ); /* drift reduction factor. <------------------------------ */
      //reg drift
      v.r   = - C*Omega*rhelio*2*(part.r-rhelio)*sin(part.th) *( (2*delta_m*delta_m*part.r*part.r + rhelio*rhelio*sin(part.th)*sin(part.th))*Vsw*Vsw*Vsw*cos(part.th)-0.5*(delta_m*delta_m*part.r*part.r*Vsw*Vsw-Omega*Omega*rhelio*rhelio*(part.r-rhelio)*(part.r-rhelio)*sin(part.th)*sin(part.th)*sin(part.th)*sin(part.th)+rhelio*rhelio*Vsw*Vsw*sin(part.th)*sin(part.th))*sin(part.th)*dV_dth  );
      v.th  = - C*Omega*rhelio*Vsw*sin(part.th)*sin(part.th)*( 2*part.r*(part.r-rhelio)*(delta_m*delta_m*part.r*Vsw*Vsw+Omega*Omega*rhelio*rhelio*(part.r-rhelio)*sin(part.th)*sin(part.th)*sin(part.th)*sin(part.th))-(4*part.r-3*rhelio)*E );
      v.phi = 2*C*Vsw                         *( -delta_m*delta_m*part.r*part.r*(delta_m*part.r+rhelio*cos(part.th))*Vsw*Vsw*Vsw + 2*delta_m*part.r*(E)*Vsw -Omega*Omega*rhelio*rhelio*(part.r-rhelio)*sin(part.th)*sin(part.th)*sin(part.th)*sin(part.th)*(delta_m*part.r*part.r*Vsw-rhelio*(part.r-rhelio)*Vsw*cos(part.th)+rhelio*(part.r-rhelio)*sin(part.th)*dV_dth  )  );
      //ns drift
      C= Vsw*Dftheta_dtheta*sin(part.th)*sin(part.th)*Ka*part.r*rhelio*rhelio/(Asun*E);
      C*= P*P/(P*P+ LIM[HZone+InitZone].P0dNS*LIM[HZone+InitZone].P0dNS );/* drift reduction factor.  <------------------------------ */
      v.r  +=   - C*Omega*sin(part.th)*(part.r-rhelio);
      //v.th += 0;
      v.phi+=  - C*Vsw   ;
  }else{
      // equatorial region. Bth = 0 
      float E = +Omega*Omega*(part.r-rhelio)*(part.r-rhelio)*sin(part.th)*sin(part.th)+Vsw*Vsw;
      float C = Omega*fth*Ka*part.r/(Asun* E*E ) ;
      C *= P*P/(P*P+ LIM[HZone+InitZone].P0d*LIM[HZone+InitZone].P0d ); /* drift reduction factor.  <------------------------------ */
      v.r   = - 2.*C*(part.r-rhelio) * ( 0.5*(Omega*Omega*(part.r-rhelio)*(part.r-rhelio)*sin(part.th)*sin(part.th)-Vsw*Vsw)*sin(part.th)*dV_dth+Vsw*Vsw*Vsw*cos(part.th) );
      v.th  = - C * Vsw*sin(part.th) * (2*Omega*Omega*part.r*(part.r-rhelio)*(part.r-rhelio)*sin(part.th)*sin(part.th) - (4*part.r-3.*rhelio)*E );
      // float Gamma=(Omega*(part.r-rhelio)*sin(part.th))/Vsw;
      // float Gamma2=Gamma*Gamma;
      // float DGamma_dr     = Omega*sin(part.th)/Vsw;
      // v.th = (Ka *part.r*fth*(Gamma*(3.+3.*Gamma2)+part.r*(1.-Gamma2)*DGamma_dr))/(Asun*(1.+Gamma2)*(1.+Gamma2));


      v.phi = 2*C*Vsw*Omega*(part.r-rhelio)*(part.r-rhelio)*sin(part.th)*(Vsw*cos(part.th)-sin(part.th)*dV_dth);
                                                                                          // if (debug){
                                                                                          //    printf("Vdr %e  vdth %e vfph %e  \n", v.r,v.th,v.phi);
                                                                                          // }
      C = Vsw*Dftheta_dtheta*Ka*part.r/(Asun* E ) ;
      C*= P*P/(P*P+ LIM[HZone+InitZone].P0dNS*LIM[HZone+InitZone].P0dNS );/* drift reduction factor.  <------------------------------ */
      v.r  += - C*Omega*(part.r-rhelio)*sin(part.th);
      v.phi+= - C*Vsw;
                                                                                          // if (debug){
                                                                                          //    printf("VdNSr %e  VdNSph %e  \n", - C*Omega*(part.r-rhelio)*sin(part.th),  - C*Vsw);
                                                                                          //    printf("Drift_PM89:: Dftheta_dtheta=%e\t KA=%e\tr=%f\tAsun=%e\tGamma2=%e\n",Dftheta_dtheta,Ka,part.r,Asun,Vsw*Vsw*Vsw*Vsw/(E*E ));
                                                                                          // }
    }


  
  // Suppression rigidity dependence: logistic function (~1 at low energy ---> ~0 at high energy)
  float HighRigiSupp=LIM[HZone+InitZone].plateau+(1.-LIM[HZone+InitZone].plateau)/(1.+exp(HighRigiSupp_smoothness*(P-HighRigiSupp_TransPoint)));
  v.r   = v.r   *HighRigiSupp;
  v.th  = v.th  *HighRigiSupp;
  v.phi = v.phi *HighRigiSupp;

  return v;
}



float EvalP0DriftSuppressionFactor(int WhichDrift,int SolarPhase, float TiltAngleDeg,float ssn){
  //WhichDrift = 0 - Regular drift
  //WhichDrift = 1 - NS drift

  float InitialVal,  FinalVal,  CenterOfTransition,  smoothness;

  if (WhichDrift==0){ // reg drift
    InitialVal = TDDS_P0d_Ini;
    FinalVal   = TDDS_P0d_Fin;
    if (SolarPhase==0 ){
      CenterOfTransition = TDDS_P0d_CoT_asc;
      smoothness         = TDDS_P0d_Smt_asc;
    }else{
      CenterOfTransition = TDDS_P0d_CoT_des;
      smoothness         = TDDS_P0d_Smt_des;      
    }
  }else{ //NS drift
    InitialVal = TDDS_P0dNS_Ini;
    FinalVal   = ssn/SSNScalF;
    if (SolarPhase==0 ){
      CenterOfTransition = TDDS_P0dNS_CoT_asc;
      smoothness         = TDDS_P0dNS_Smt_asc;
    }else{
      CenterOfTransition = TDDS_P0dNS_CoT_des;
      smoothness         = TDDS_P0dNS_Smt_des;      
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
  return SmoothTransition(InitialVal,  FinalVal,  CenterOfTransition,  smoothness,  TiltAngleDeg);
}


float EvalHighRigidityDriftSuppression_plateau(int SolarPhase, float TiltAngleDeg){
  // Plateau time dependence
  float CenterOfTransition,  smoothness;
  if (SolarPhase==0 ){
    CenterOfTransition = HRS_TransPoint_asc;
    smoothness         = HRS_smoothness_asc;
  }else{
    CenterOfTransition = HRS_TransPoint_des;
    smoothness         = HRS_smoothness_des;      
  }
  return 1.-SmoothTransition(1.,  0.,  CenterOfTransition,  smoothness,  TiltAngleDeg);
}


