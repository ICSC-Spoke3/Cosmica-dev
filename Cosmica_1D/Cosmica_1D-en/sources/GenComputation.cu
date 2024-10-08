#include "GenComputation.cuh"
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"

#include <math.h>           // c math library

////////////////////////////////////////////////////////////////
//..... GEneric useful and safe function .......................
////////////////////////////////////////////////////////////////

int ceil_int(int a, int b){
  // https://www.reddit.com/r/C_Programming/comments/gqpuef/comment/fru7tmu/?utm_source=share&utm_medium=web2x&context=3
  return ((a+(b-1))/b);
}

int floor_int(int a, int b){
  return int(floor(a/b));
}

__device__ float sign(float num){
  if (num>=0) return 1;
  else return -1;
}

__device__ float SmoothTransition(float InitialVal, float FinalVal, float CenterOfTransition, float smoothness, float x) {
  if (smoothness==0) {
    if (x>=CenterOfTransition) return FinalVal;
    else                       return InitialVal;
  }
  
  else return (InitialVal+FinalVal)/2.-(InitialVal-FinalVal)/2.*tanh((x-CenterOfTransition)/smoothness);
}

__device__ float beta_(float T, float T0) {
  return sqrt(T*(T + T0 + T0))/(T + T0);
}

__device__ float Rigidity(float T, struct PartDescription_t part) {
  return part.A/fabs(part.Z)*sqrt(T*(T+2.*part.T0));
}
