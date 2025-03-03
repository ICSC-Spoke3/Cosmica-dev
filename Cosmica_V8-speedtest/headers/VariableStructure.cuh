#ifndef VariableStructure
#define VariableStructure
#include <GenComputation.cuh>
#include <HeliosphereModel.cuh>
#include <constants.hpp>

/**
 * @brief Struct with the structure of the heliospheric magnetic field
 */
struct LaunchParam_t {
    unsigned blocks = 0, threads = 0;

    /**
     * @brief Computes the launch parameters based on threads per block (TpB) and total parts (NParts).
     * @param TpB Threads per block.
     * @param NParts Total number of computational parts.
     * @return Computed LaunchParam_t with optimal blocks and threads.
     */
    static LaunchParam_t from_TpB(const unsigned TpB, const unsigned NParts) {
        return {(NParts + TpB - 1) / TpB, TpB};
    }
};

/**
 * @brief Struct with the structure of the heliospheric magnetic field
 */
struct InputHeliosphericParametrizationProperties_t {
    float k0 = 0;
};

/**
 * @brief Struct with the structure of the heliospheric magnetic field
 */
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

/**
 * @brief Struct with heliosheat physical parameters
 */
struct InputHeliosheatParameters_t {
    float k0 = 0;
    float V0 = 0; // solar wind at termination shock
};

/**
 * @brief Struct of initial propagation positions arrays with coordinates and rigidities
 */
struct InitialPositions_t {
    float *r; // heliocentric radial distances
    float *th; // heliocentric polar angles
    float *phi; // heliocentric azimutal - longitudinal angles (really needed in 2D model?)
};

/**
 * @brief Struct of the quasi-particle with coordinates, rigidities and time of flight
 */
struct QuasiParticle_t {
    float r; // heliocentric radial distances
    float th; // heliocentric polar angles
    float phi; // heliocentric azimutal - longitudinal angles (really needed in 2D model?)
    float R; // rigidity (GeV/n?)
    float t_fly;

    /**
     * @brief Normalize the angles to the correct range.
     */
    __forceinline__ __device__ void normalize_angles() {
        th = fabsf(th);
        th = fabsf(fmodf(2 * Pi + safeSign(Pi - th) * th, Pi));
        th = 2 * clamp(th, thetaNorthlimit, thetaSouthlimit) - th;

        phi = fmodf(phi, 2 * Pi);
        phi = fmodf(2 * Pi + phi, 2 * Pi);
    }
};

/**
 * @brief Struct of threads quasi-particles with coordinates, rigidities and time of flight
 */
struct ThreadQuasiParticles_t {
    float *r; // heliocentric radial distances
    float *th; // heliocentric polar angles
    float *phi; // heliocentric azimutal - longitudinal angles (really needed in 2D model?)
    float *R; // rigidity (GeV/n?)
    float *t_fly; // total propagation time
    // float* alphapath; // Montecarlo statistical weight - exponent of c factor

    /**
     * @brief Get the quasi-particle at the given index.
     * @param id Index of the quasi-particle.
     * @return QuasiParticle_t with the coordinates, rigidities and time of flight.
     */
    __forceinline__ __device__ QuasiParticle_t get(const unsigned id) const {
        return {r[id], th[id], phi[id], R[id], t_fly[id]};
    }
};

/**
 * @brief Struct of the index of the quasi-particle in the grid
 */
struct Index_t {
    const unsigned param, isotope, period;
    int radial = 0;

    /**
     * @brief Update the index with the given quasi-particle.
     * @param qp QuasiParticle_t with the coordinates, rigidities and time of flight.
     */
    __forceinline__ __device__ void update(const QuasiParticle_t &qp) {
        radial = RadialZone(period, qp);
    }

    /**
     * @brief Get the combined index of the quasi-particle.
     * @return Combined index of the quasi-particle.
     */
    __forceinline__ __device__ unsigned combined() const {
        return period + radial;
    }

    /**
     * @brief Get the instance index of the quasi-particle.
     * @param NIsotopes Number of isotopes.
     * @return Instance index of the quasi-particle.
     */
    __forceinline__ __device__ unsigned instance(const unsigned NIsotopes) const {
        return param * NIsotopes + isotope;
    }
};

/**
 * @brief Struct of the index of the quasi-particle in the grid
 */
struct ThreadIndexes_t {
    unsigned size;
    unsigned *param, *isotope, *period;

    /**
     * @brief Get the index at the given id.
     * @param id Index of the index.
     * @return Index_t with the combined index of the quasi-particle.
     */
    __forceinline__ __host__ __device__ Index_t get(const unsigned id) const {
        return {param[id], isotope[id], period[id]};
    }
};

// TODO: check if we can use the matrix cuda optimized libraries
/**
 * @brief Struct with the structure of square root decomposition of symmetric difusion tensor
 */
struct Tensor3D_t {
    float rr = 0;
    float tr = 0;
    float tt = 0;
    float pr = 0;
    float pt = 0;
    float pp = 0; // not null components
};

/**
 * @brief Struct with the structure of the the symmetric difusion tensor and its derivative
 */
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

/**
 * @brief Struct with the structure of the advective-drift vector
 */
struct vect3D_t {
    float r = 0; // heliocentric radial component
    float th = 0; // heliocentric polar component
    float phi = 0.; // heliocentric azimutal - longitudinal angle component
};

/**
 * @brief Struct with the structure of the output result of a single energy simulation
 */
struct MonteCarloResult_t {
    unsigned Nregistered;
    int Nbins;
    float LogBin0_lowEdge; // lower boundary of first bin
    float DeltaLogR; // Bin amplitude in log scale
    float *BoundaryDistribution;
};

typedef MonteCarloResult_t* InstanceHistograms;

#endif
