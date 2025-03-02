#ifndef VariableStructure
#define VariableStructure
#include <GenComputation.cuh>
#include <HeliosphereModel.cuh>
#include <constants.hpp>

// Struct with threads, blocks and share memory with which launch a cuda function
struct LaunchParam_t {
    unsigned threads = 0;
    unsigned blocks = 0;
};

struct InputHeliosphericParametrizationProperties_t {
    float k0 = 0;
};

struct InputHeliosphericProperties_t {
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
};

// Heliosheat physical parameters
struct InputHeliosheatParameters_t {
    float k0 = 0;
    float V0 = 0; // solar wind at termination shock
};

// Struct of initial propagation positions arrays with coordinates and rigidities
struct InitialPositions_t {
    float *r; // heliocentric radial distances
    float *th; // heliocentric polar angles
    float *phi; // heliocentric azimutal - longitudinal angles (really needed in 2D model?)
};

struct QuasiParticle_t {
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
};

// Struct of quasi-particles arrays with coordinates, rigidities and time of flight
struct ThreadQuasiParticles_t {
    float *r; // heliocentric radial distances
    float *th; // heliocentric polar angles
    float *phi; // heliocentric azimutal - longitudinal angles (really needed in 2D model?)
    float *R; // rigidity (GeV/n?)
    float *t_fly; // total propagation time
    // float* alphapath; // Montecarlo statistical weight - exponent of c factor

    __forceinline__ __device__ QuasiParticle_t get(const unsigned id) const {
        return {r[id], th[id], phi[id], R[id], t_fly[id]};
    }
};

struct Index_t {
    const unsigned param, isotope, period;
    int radial = 0;

    __forceinline__ __device__ void update(const QuasiParticle_t &qp) {
        radial = RadialZone(period, qp);
    }

    __forceinline__ __device__ unsigned combined() const {
        return period + radial;
    }

    __forceinline__ __device__ unsigned instance(const unsigned NIsotopes) const {
        return param * NIsotopes + isotope;
    }
};

struct ThreadIndexes_t {
    unsigned size;
    unsigned *param, *isotope, *period;
    __forceinline__ __host__ __device__ Index_t get(const unsigned id) const {
        return {param[id], isotope[id], period[id]};
    }
};

// SEE IF WE CAN USE THE MATRIX CUDA UPTIMIZED LIBRARIES
// Struct with the structure of square root decomposition of symmetric difusion tensor
struct Tensor3D_t {
    float rr = 0;
    float tr = 0;
    float tt = 0;
    float pr = 0;
    float pt = 0;
    float pp = 0; // not null components
};

// Struct with the structure of the the symmetric difusion tensor and its derivative
struct DiffusionTensor_t {
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
};

// Struct with the structure of the advective-drift vector
struct vect3D_t {
    float r = 0; // heliocentric radial component
    float th = 0; // heliocentric polar component
    float phi = 0.; // heliocentric azimutal - longitudinal angle component
};

// Data container for output result of a single energy simulation
struct MonteCarloResult_t {
    unsigned Nregistered;
    int Nbins;
    float LogBin0_lowEdge; // lower boundary of first bin
    float DeltaLogR; // Bin amplitude in log scale
    float *BoundaryDistribution;
};

typedef MonteCarloResult_t* InstanceHistograms;

#endif
