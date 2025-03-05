#ifndef HelModVariableStructure
#define HelModVariableStructure
#include <vector>
#include <string>

#include "VariableStructure.cuh"

////////////////////////////////////////////////////////////////
// Definition of the tuned model quantities and function parameters
////////////////////////////////////////////////////////////////

struct PartDescription_t {
    float T0 = 0.938; // rest mass in GeV/n
    float Z = 1; // Atomic number
    float A = 1; // mass number
}; // particle type

struct HeliosphereBoundRadius_t {
    float Rts_nose = 100; // Termination shock position
    float Rhp_nose = 122; // Heliopause position
    float Rts_tail = 100; // Termination shock position
    float Rhp_tail = 122; // Heliopause position
};

struct HeliosphereParametrizationProperties_t {
    float k0_paral[2] = {};
    float k0_perp[2] = {};
    float GaussVar[2] = {};
};

struct HeliosphereProperties_t {
    float V0 = 0; // [AU/s] Radial Solar wind speed on ecliptic
    float g_low = 0; // for evaluating Kpar, glow parameter
    float rconst = 0; // for evaluating Kpar, rconst parameter
    float TiltAngle = 0; // Tilt angle of neutral sheet
    float Asun = 0; // normalization constant of HMF
    float P0d = 0; // Drift Suppression rigidity
    float P0dNS = 0; // NS Drift Suppression rigidity
    float plateau = 0.; // Time dependent plateau in the high rigidity suppression
};

struct HeliosheatProperties_t {
    // properties related to heliopause
    float V0 = 0; // [AU/s] Radial Solar wind speed on ecliptic
    float k0 = 0; // Diffusion parameter for parallel term of diffusion tensor
};


struct SimulationParametrizations_t {
    struct Parametrization_t {
        HeliosphereParametrizationProperties_t heliosphere[NMaxRegions];
    };

    unsigned Nparams;
    Parametrization_t *params;
};

struct SimulationConstants_t {
    float Min_dt = MIN_DT, Max_dt = MAX_DT, Timeout = TIMEOUT;

    unsigned NIsotopes = 0;
    PartDescription_t Isotopes[NMaxIsotopes];

    unsigned Nregions = 0; // Number of Inner Heliosphere region (15 inner region + 1 Heliosheat)
    HeliosphereBoundRadius_t RadBoundary_effe[NMaxRegions] = {}; // boundaries in effective heliosphere
    HeliosphereBoundRadius_t RadBoundary_real[NMaxRegions] = {}; // real boundaries heliosphere
    bool IsHighActivityPeriod[NMaxRegions] = {false}; // active the modification for high activity period

    HeliosphereProperties_t heliosphere_properties[NMaxRegions];
    HeliosheatProperties_t heliosheat_properties[NMaxRegions];
};

struct SimConfiguration_t {
    std::string output_file;
    unsigned long RandomSeed = 0;
    unsigned Npart = 5000; // number of event to be simulated
    unsigned NT; // number of bins of energies to be simulated
    unsigned NInitialPositions = 0;
    float *Tcentr; // array of energies to be simulated
    float RelativeBinAmplitude = 0.00855;
    InitialPositions_t InitialPositions; // initial position
    InstanceHistograms *Results; // output of the code
    // SimulatedHeliosphere_t HeliosphereToBeSimulated; // Heliosphere properties for the simulation
    SimulationParametrizations_t simulation_parametrization;
    SimulationConstants_t simulation_constants;
    std::vector<std::string> isotopes_names;
};

#include <spdlog/common.h>

inline std::istream& operator>>(std::istream& is, spdlog::level::level_enum& level) {
    std::string token;
    is >> token;
    level = spdlog::level::from_str(token);  // Convert string to spdlog::level::level_enum
    return is;
}


#endif


