#include "GenComputation.cuh"
#include "HelModVariableStructure.cuh"
#include "VariableStructure.cuh"
#include "MagneticDrift.cuh"
#include "SolarWind.cuh"


#include <stdio.h>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions


////////////////////////////////////////////////////////////////
//..... Magnetic Drift Model ..................................  
//.... Potgieter Mooral 1985 - See Burger&Hatttingh 1995
////////////////////////////////////////////////////////////////
__device__ float Gamma_Bfield(float r, float th, float Vsw) {
  return (Omega*(r-rhelio)*sin(th))/Vsw;
}
__device__ float delta_Bfield(float r, float th) {
  return r/rhelio*delta_m/sin(th);;
}

__device__ float eval_Ka(PartDescription_t pt, float R) {

  return (sign(pt.Z)*(GeV)*beta_R(R,pt))*R/3;  /* sign(Z)*beta*P/3 constant part of antisymmetric diffusion coefficient */
}

// .. Scaling factor of drift in 2D approximation.  to account Neutral sheet
/* scaling function of drift vel */
__device__ float eval_fth(float th, float theta_mez) {

  if (theta_mez<(Pi/.2)) {
    return 1./(acos(Pi/(2.*theta_mez)-1))*atan((1.-(2.*th/Pi))*tan(acos(Pi/(2.*theta_mez)-1)));
  }
  
  else {/* if the smoothness parameter "theta_mez" is greater then Pi/2, then the neutral sheet is flat, only heaviside function is applied.*/
    if (th>Pi/2) return 1.;
    if (th<Pi/2) return -1.;
    if (th==Pi/2) return 0.;
  }
}

__device__ float eval_Dftheta_dtheta(float th, float theta_mez) {

  if (theta_mez<(Pi/.2)){
    float a_f      = acos(Pi/(2.*theta_mez)-1);
    return - 2. * tan(a_f)/( a_f*Pi*(1.+(1-2.*th/Pi)*(1-2.*th/Pi)* tan(a_f)* tan(a_f) ));
  }
  
  else return 0.;
}

// Suppression rigidity dependence: logistic function (~1 at low energy ---> ~0 at high energy)

__device__ float eval_HighRigiSupp(int InitZone, int HZone, float R) {
  return LIM[HZone+InitZone].plateau+(1.-LIM[HZone+InitZone].plateau)/(1.+exp(HighRigiSupp_smoothness*(R-HighRigiSupp_TransPoint)));
}

__device__ float eval_E_drift(bool IsPolarRegion, float r, float th, float Vsw) {

  if(IsPolarRegion) return delta_m*delta_m*r*r*Vsw*Vsw+Omega*Omega*rhelio*rhelio*(r-rhelio)*(r-rhelio)*sin(th)*sin(th)*sin(th)*sin(th)+rhelio*rhelio*Vsw*Vsw*sin(th)*sin(th);
  else return Omega*Omega*(r-rhelio)*(r-rhelio)*sin(th)*sin(th)+Vsw*Vsw;
}

__device__ float eval_C_drift_reg(float Asun, float IsPolarRegion, float r, float th, float Ka, int InitZone, int HZone, float R, float fth, float E){
  //reg drift
  float C = (IsPolarRegion)? fth*sin(th)*Ka*r*rhelio/(Asun*E*E): Omega*fth*Ka*r/(Asun*E*E);
  C *= R*R/(R*R+ LIM[HZone+InitZone].P0d*LIM[HZone+InitZone].P0d ); /* drift reduction factor. <------------------------------ */
}

__device__ float eval_C_drift_ns(float Asun, float IsPolarRegion, float r, float th, float Ka, int InitZone, int HZone, float R,float Vsw, float Dftheta_dtheta, float E){
  //ns drift
  float C = (IsPolarRegion)? Vsw*Dftheta_dtheta*sin(th)*sin(th)*Ka*r*rhelio*rhelio/(Asun*E): Vsw*Dftheta_dtheta*Ka*r/(Asun*E);
  C *= R*R/(R*R+ LIM[HZone+InitZone].P0dNS*LIM[HZone+InitZone].P0dNS );/* drift reduction factor.  <------------------------------ */
}

__device__ vect3D_t Drift_PM89(unsigned char InitZone, signed char HZone, float r, float th, float phi, float R, bool IsPolarRegion, float Asun, float Ka, float fth, float Dftheta_dtheta, float Vsw, float dV_dth, float HighRigiSupp){
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

  if (IsPolarRegion){
    // polar region
    float E = eval_E_drift(true, r, th, Vsw);
    //reg drift
    float C = eval_C_drift_reg(Asun, true, r, th, Ka, InitZone, HZone, R, fth, E);
    v.r   = - C*Omega*rhelio*2*(r-rhelio)*sin(th) *( (2*delta_m*delta_m*r*r + rhelio*rhelio*sin(th)*sin(th))*Vsw*Vsw*Vsw*cos(th)-0.5*(delta_m*delta_m*r*r*Vsw*Vsw-Omega*Omega*rhelio*rhelio*(r-rhelio)*(r-rhelio)*sin(th)*sin(th)*sin(th)*sin(th)+rhelio*rhelio*Vsw*Vsw*sin(th)*sin(th))*sin(th)*dV_dth  );
    v.th  = - C*Omega*rhelio*Vsw*sin(th)*sin(th)*( 2*r*(r-rhelio)*(delta_m*delta_m*r*Vsw*Vsw+Omega*Omega*rhelio*rhelio*(r-rhelio)*sin(th)*sin(th)*sin(th)*sin(th))-(4*r-3*rhelio)*E );
    v.phi = 2*C*Vsw                         *( -delta_m*delta_m*r*r*(delta_m*r+rhelio*cos(th))*Vsw*Vsw*Vsw + 2*delta_m*r*(E)*Vsw -Omega*Omega*rhelio*rhelio*(r-rhelio)*sin(th)*sin(th)*sin(th)*sin(th)*(delta_m*r*r*Vsw-rhelio*(r-rhelio)*Vsw*cos(th)+rhelio*(r-rhelio)*sin(th)*dV_dth  )  );
    //ns drift
    C = eval_C_drift_ns(Asun, true, r, th, Ka, InitZone, HZone, R,Vsw, Dftheta_dtheta, E);
    v.r  +=   - C*Omega*sin(th)*(r-rhelio);
    v.th  = - C * Vsw*sin(th) * (2*Omega*Omega*r*(r-rhelio)*(r-rhelio)*sin(th)*sin(th) - (4*r-3.*rhelio)*E );
    v.phi+=  - C*Vsw   ;
  }
  else{
    // equatorial region. Bth = 0 
    float E = eval_E_drift(false, r, th, Vsw);
    //reg drift
    float C = eval_C_drift_reg(Asun, false, r, th, Ka, InitZone, HZone, R, fth, E);
    v.r   = - 2.*C*(r-rhelio) * ( 0.5*(Omega*Omega*(r-rhelio)*(r-rhelio)*sin(th)*sin(th)-Vsw*Vsw)*sin(th)*dV_dth+Vsw*Vsw*Vsw*cos(th) );
    v.phi = 2*C*Vsw*Omega*(r-rhelio)*(r-rhelio)*sin(th)*(Vsw*cos(th)-sin(th)*dV_dth);
    //ns drift
    C = eval_C_drift_ns(Asun, false, r, th, Ka, InitZone, HZone, R,Vsw, Dftheta_dtheta, E);
    v.r  += - C*Omega*(r-rhelio)*sin(th);
    v.phi+= - C*Vsw;
  }

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
    }
    else{
      CenterOfTransition = TDDS_P0d_CoT_des;
      smoothness         = TDDS_P0d_Smt_des;      
    }
  }
  else{ //NS drift
    InitialVal = TDDS_P0dNS_Ini;
    FinalVal   = ssn/SSNScalF;
    if (SolarPhase==0 ){
      CenterOfTransition = TDDS_P0dNS_CoT_asc;
      smoothness         = TDDS_P0dNS_Smt_asc;
    }
    else{
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
  }
  else{
    CenterOfTransition = HRS_TransPoint_des;
    smoothness         = HRS_smoothness_des;
  }
  return 1.-SmoothTransition(1.,  0.,  CenterOfTransition,  smoothness,  TiltAngleDeg);
}