#ifndef SolarWind
#define SolarWind
// ------------------------------------------
// solar wind at termination shock tuned parameters
#ifndef s_tl
  #define s_tl 2.7   
#endif
#ifndef L_tl
  #define L_tl 0.09    // smooting distance for computing the decrease of solar wind
#endif
// ------------------------------------------
// solar wind latitudinal structure parameters
#ifndef Vhigh
  #define Vhigh 760./aukm   
#endif
// ------------------------------------------
__device__ float SolarWindSpeed(unsigned short ,signed short, qvect_t ); 
__device__ float DerivativeOfSolarWindSpeed_dtheta(unsigned short ,signed short, qvect_t ); 
// ------------------------------------------
#endif
