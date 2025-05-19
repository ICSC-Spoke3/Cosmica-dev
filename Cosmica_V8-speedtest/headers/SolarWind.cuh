#ifndef SolarWind
#define SolarWind
// Solar wind at termination shock - tuned parameters
#ifndef s_tl
#define s_tl 2.7f
#endif
#ifndef L_tl
#define L_tl 0.09f  // Smoothing distance for computing the decrease of solar wind
#endif

// Solar wind latitudinal structure parameters
#ifndef Vhigh
#define Vhigh 760.f/aukm
#endif

__device__ float SolarWindSpeed(const Index_t &, const QuasiParticle_t &);

__device__ float DerivativeOfSolarWindSpeed_dtheta(const Index_t &, const QuasiParticle_t &);

#endif
