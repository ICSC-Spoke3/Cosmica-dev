#include "globals.h"

__device__ float SmoothTransition(float InitialVal, float FinalVal, float CenterOfTransition, float smoothness, float x)
{ // smooth transition between  InitialVal to FinalVal centered at CenterOfTransition as function of x
  // if smoothness== 0 use a sharp transition
  if (smoothness==0)
  {
    if (x>=CenterOfTransition) return FinalVal;
    else                       return InitialVal;
  }
  else 
  {return (InitialVal+FinalVal)/2.-(InitialVal-FinalVal)/2.*tanh((x-CenterOfTransition)/smoothness);}
}

__device__ float beta_(float T, float T0) {
  // beta value =v/c from kinetic energy
  return sqrt(T*(T + T0 + T0))/(T + T0);
}

__device__ float Rigidity(float T,PartDescription_t part) {
  // convert Kinetic Energy in Rigidity
  return part.A/fabs(part.Z)*sqrt(T*(T+2.*part.T0));
}

// __device__ Tensor3D_t initTensor3D_t(){
//   Tensor3D_t T3D;
//   T3D.rr=0;
//   T3D.tr=0;
//   T3D.pr=0;
//   T3D.tt=0;
//   T3D.pt=0;
//   T3D.pp=0;
//   return T3D;
// }