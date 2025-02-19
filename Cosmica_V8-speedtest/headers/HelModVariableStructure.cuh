#ifndef HelModVariableStructure
#define HelModVariableStructure
#include <stdio.h>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions
#include "VariableStructure.cuh"

////////////////////////////////////////////////////////////////
// Definition of the tuned model quantities and function parameters
////////////////////////////////////////////////////////////////

#define Pi      3.141592653589793f      // Pi
#define Half_Pi 1.570796326794896f      // Pi/2
#define aukm    149597870.691f          // 1 AU in km precise is 149.597.870,691 Km = 1e8 Km
#define aum     149597870691.f          // 1 AU in m precise is 149.597.870.691  m  = 1e11 m
#define aucm    14959787069100.f        // 1 AU in m precise is 149.597.870.691.00  cm  = 1e13 m
#define MeV     1e6f                    // MeV->eV                               eV
#define GeV     1e9f                    // MeV->eV                               eV
#define c       3e8f                    // Light Velodity                        m/s
#define thetaNorthlimit 0.000010f       // maximun value of latitude at north pole (this to esclude the region of not physical results due to divergence of equations)


#define thetaSouthlimit Pi-thetaNorthlimit // maximun value of latitude at south pole (this to esclude the region of not physical results due to divergence of equations)


#define struct_string_lengh 70
#define MaxCharinFileName   90
#define ReadingStringLenght 2000        // max lenght of each row while reading input file
//emulate bool
#define True    1
#define False   0

#define VERBOSE_low  1
#define VERBOSE_med  2
#define VERBOSE_hig  3

#ifndef PolarZone
#define PolarZone 30
#endif
#define CosPolarZone cosf(PolarZone*Pi/180.f)
#ifndef delta_m
#define delta_m 2.000000e-05f       // megnetic field disturbance in high latitude region
#endif
#ifndef TiltL_MaxActivity_threshold
#define TiltL_MaxActivity_threshold 50
#endif

// Solar param constant
#define Omega  3.03008e-6f  // solar angular velocity
#define rhelio 0.004633333f  // solar radius in AU


typedef struct options_t {
    unsigned char verbose;
    FILE *input;
} options_t;

typedef struct PartDescription_t {
    float T0 = 0.938; // rest mass in GeV/n
    float Z = 1; // Atomic number
    float A = 1; // mass number
} PartDescription_t; // particle type

typedef struct HeliosphereBoundRadius_t {
    float Rts_nose = 100; // Termination shock position
    float Rhp_nose = 122; // Heliopause position
    float Rts_tail = 100; // Termination shock position
    float Rhp_tail = 122; // Heliopause position
} HeliosphereBoundRadius_t;


#define NMaxRegions 335    // about 25 year of simulations

typedef struct HeliosphereZoneProperties_t {
    // properties related to heliosphere like dimension, coefficients
    float V0 = 0; // [AU/s] Radial Solar wind speed on ecliptic
    float k0_paral[2] = {0};
    // Diffusion parameter for parallel term of diffusion tensor [for HighActivity, for Low activity]
    float k0_perp[2] = {0};
    // Diffusion parameter for perpendicular term of diffusion tensor  [for HighActivity, for Low activity]
    float GaussVar[2] = {0}; // Gaussian variation for Diffusion parameter[for HighActivity, for Low activity]
    float g_low = 0; // for evaluating Kpar, glow parameter
    float rconst = 0; // for evaluating Kpar, rconst parameter
    float TiltAngle = 0; // Tilt angle of neutral sheet
    float Asun = 0; // normalization constant of HMF
    float P0d = 0; // Drift Suppression rigidity
    float P0dNS = 0; // NS Drift Suppression rigidity
    float plateau = 0.; // Time dependent plateau in the high rigidity suppression
    signed int Polarity = 0; // HMF polarity
} HeliosphereZoneProperties_t;

typedef struct HeliosheatProperties_t {
    // properties related to heliopause
    float V0 = 0; // [AU/s] Radial Solar wind speed on ecliptic
    float k0 = 0; // Diffusion parameter for parallel term of diffusion tensor
} HeliosheatProperties_t;

typedef struct SimulatedHeliosphere_t {
    // properties related to heliosphere like dimension, coefficients
    float Rmirror = 0.3; // [AU] Internal heliosphere bounduary - mirror radius.
    unsigned int Nregions = 0; // Number of Inner Heliosphere region (15 inner region + 1 Heliosheat)
    HeliosphereBoundRadius_t RadBoundary_effe[NMaxRegions] = {0}; // boundaries in effective heliosphere
    HeliosphereBoundRadius_t RadBoundary_real[NMaxRegions] = {0}; // real boundaries heliosphere
    bool IsHighActivityPeriod[NMaxRegions] = {false}; // active the modification for high activity period
    //  HeliosphereZoneProperties_t prop_medium[NMaxRegions];             // PROPerties of the interplanetary MEDIUM - Heliospheric Parameters in each Heliospheric Zone
} SimulatedHeliosphere_t;

typedef struct SimParameters_t {
    // Place here all simulation variables
    char output_file_name[struct_string_lengh] = "SimTest";
    unsigned long RandomSeed = 0;
    unsigned int Npart = 5000; // number of event to be simulated
    unsigned int NT; // number of bins of energies to be simulated
    unsigned int NInitialPositions = 0;
    // number of initial positions -> this number represent also the number of Carrington rotation that
    float *Tcentr; // array of energies to be simulated
    vect3D_t *InitialPosition; // initial position
    PartDescription_t IonToBeSimulated; // Ion to be simulated
    MonteCarloResult_t *Results; // output of the code
    float RelativeBinAmplitude = 0.00855;
    // relative (respect 1.) amplitude of Energy bin used as X axis in BoundaryDistribution  --> delta T = T*RelativeBinAmplitude
    SimulatedHeliosphere_t HeliosphereToBeSimulated; // Heliosphere properties for the simulation
    HeliosphereZoneProperties_t prop_medium[NMaxRegions];
    // PROPerties of the interplanetary MEDIUM - Heliospheric Parameters in each Heliospheric Zone
    HeliosheatProperties_t prop_Heliosheat[NMaxRegions]; // Properties of Heliosheat
} SimParameters_t;

#endif

#ifndef MAINCU
// -----------------------------------------------------------------
// ------------  Device Constant Variables declaration -------------
// -----------------------------------------------------------------
extern __constant__ SimulatedHeliosphere_t Heliosphere;
// Heliosphere properties include Local Interplanetary medium parameters
extern __constant__ HeliosphereZoneProperties_t LIM[NMaxRegions];
extern __constant__ HeliosheatProperties_t HS[NMaxRegions]; // heliosheat
#endif
