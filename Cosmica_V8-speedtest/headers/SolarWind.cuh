#ifndef SolarWind
#define SolarWind
// ------------------------------------------
// solar wind at termination shock tuned parameters
#ifndef s_tl
#define s_tl 2.7f
#endif
#ifndef L_tl
#define L_tl 0.09f    // smooting distance for computing the decrease of solar wind
#endif
// ------------------------------------------
// solar wind latitudinal structure parameters
#ifndef Vhigh
#define Vhigh 760.f/aukm
#endif
// ------------------------------------------
__device__ float SolarWindSpeed(const Index_t &, float, float, float, const HeliosphereZoneProperties_t *LIM);

__device__ float DerivativeOfSolarWindSpeed_dtheta(const Index_t &, float, float, float, const HeliosphereZoneProperties_t *LIM);

// ------------------------------------------
#endif
