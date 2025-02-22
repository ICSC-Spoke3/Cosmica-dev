#ifndef VariableStructure
#define VariableStructure
#include <GenComputation.cuh>
#include <HeliosphereModel.cuh>
#include <constants.hpp>

// Struct with threads, blocks and share memory with which launch a cuda function
typedef struct LaunchParam_t {
    int threads = 0;
    int blocks = 0;
    int smem = 0;
} LaunchParam_t;

// Heliospheric physical parameters
typedef struct InputHeliosphericParameters_t {
    float k0 = 0;
    float ssn = 0;
    float V0 = 0;
    float TiltAngle = 0;
    float SmoothTilt = 0;
    float BEarth = 0;
    int Polarity = 0;
    int SolarPhase = 0;
    float NMCR = 0;
    float Rts_nose = 0;
    float Rts_tail = 0;
    float Rhp_nose = 0;
    float Rhp_tail = 0;
} InputHeliosphericParameters_t;

// Heliosheat physical parameters
typedef struct InputHeliosheatParameters_t {
    float k0 = 0;
    float V0 = 0; // solar wind at termination shock
} InputHeliosheatParameters_t;

// Struct of initial propagation positions arrays with coordinates and rigidities
typedef struct InitialPositions_t {
    float *r; // heliocentric radial distances
    float *th; // heliocentric polar angles
    float *phi; // heliocentric azimutal - longitudinal angles (really needed in 2D model?)
} InitialPositions_t;

typedef struct QuasiParticle_t {
    float r; // heliocentric radial distances
    float th; // heliocentric polar angles
    float phi; // heliocentric azimutal - longitudinal angles (really needed in 2D model?)
    float R; // rigidity (GeV/n?)
    float t_fly;

    __forceinline__ __device__ void normalize_angles() {
        th = fabsf(th);
        th = fabsf(fmodf(2 * Pi + safeSign(Pi - th) * th, Pi));
        th = 2 * clamp(th, thetaNorthlimit, thetaSouthlimit) - th;

        phi = fmodf(phi, 2 * Pi);
        phi = fmodf(2 * Pi + phi, 2 * Pi);
    }
} QuasiParticle_t;

// Struct of quasi-particles arrays with coordinates, rigidities and time of flight
typedef struct ThreadQuasiParticles_t {
    float *r; // heliocentric radial distances
    float *th; // heliocentric polar angles
    float *phi; // heliocentric azimutal - longitudinal angles (really needed in 2D model?)
    float *R; // rigidity (GeV/n?)
    float *t_fly; // total propagation time
    // float* alphapath; // Montecarlo statistical weight - exponent of c factor

    __forceinline__ __device__ QuasiParticle_t get(const unsigned id) const {
        return {r[id], th[id], phi[id], R[id], t_fly[id]};
    }
} ThreadQuasiParticles_t;

typedef struct Index_t {
    const unsigned simulation, period, particle;
    int radial = 0;

    __forceinline__ __device__ void update(const QuasiParticle_t &qp) {
        radial = RadialZone(period, qp);
    }

    __forceinline__ __device__ unsigned combined() const {
        return period + radial;
    }
} Index_t;

typedef struct ThreadIndexes_t {
    unsigned *simulation, *period, *particle;
    __forceinline__ __host__ __device__ Index_t get(const unsigned id) const {
        return {simulation[id], period[id], particle[id]};
    }
} ThreadIndexes_t;

// SEE IF WE CAN USE THE MATRIX CUDA UPTIMIZED LIBRARIES
// Struct with the structure of square root decomposition of symmetric difusion tensor
typedef struct Tensor3D_t {
    float rr = 0;
    float tr = 0;
    float tt = 0;
    float pr = 0;
    float pt = 0;
    float pp = 0; // not null components
} Tensor3D_t;

// Struct with the structure of the the symmetric difusion tensor and its derivative
typedef struct DiffusionTensor_t {
    float rr = 0;
    float tr = 0;
    float tt = 0;
    float pr = 0;
    float pt = 0;
    float pp = 0;
    float DKrr_dr = 0;
    float DKtr_dt = 0;
    float DKrt_dr = 0;
    float DKtt_dt = 0;
    float DKrp_dr = 0;
    float DKtp_dt = 0; // not null components
} DiffusionTensor_t;

// Struct with the structure of the advective-drift vector
typedef struct vect3D_t {
    float r = 0; // heliocentric radial component
    float th = 0; // heliocentric polar component
    float phi = 0.; // heliocentric azimutal - longitudinal angle component
} vect3D_t;

// Data container for output result of a single energy simulation
typedef struct MonteCarloResult_t {
    unsigned long Nregistered;
    int Nbins;
    float LogBin0_lowEdge; // lower boundary of first bin
    float DeltaLogR; // Bin amplitude in log scale
    float *BoundaryDistribution;
} MonteCarloResult_t;

#endif
