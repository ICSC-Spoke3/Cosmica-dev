#ifndef GLOBALS
#define GLOBALS

#define Pi      3.141592653589793      // Pi
#define Half_Pi 1.570796326794896      // Pi/2
#define aukm    149597870.691          // 1 AU in km precise is 149.597.870,691 Km = 1e8 Km
#define aum     149597870691.          // 1 AU in m precise is 149.597.870.691  m  = 1e11 m
#define aucm    14959787069100.        // 1 AU in m precise is 149.597.870.691.00  cm  = 1e13 m
#define MeV     1e6                    // MeV->eV                               eV
#define GeV     1e9                    // MeV->eV                               eV
#define c       3e8                    // Light Velodity                        m/s
#define thetaNorthlimit 0.000010       // maximun value of latitude at north pole (this to esclude the region of not physical results due to divergence of equations)
#define thetaSouthlimit Pi-thetaNorthlimit // maximun value of latitude at south pole (this to esclude the region of not physical results due to divergence of equations)
#define struct_string_lengh 90
#define MaxCharinFileName   110 
//emulate bool
#define True    1
#define False   0

#define VERBOSE_low  1
#define VERBOSE_med  2
#define VERBOSE_hig  3

#ifndef PolarZone
 #define PolarZone 30
#endif
#define CosPolarZone cos(PolarZone*Pi/180.)
#ifndef delta_m
 #define delta_m 2.000000e-05       // megnetic field disturbance in high latitude region
#endif
#ifndef TiltL_MaxActivity_threshold
 #define TiltL_MaxActivity_threshold 50
#endif

// Solar param constant
#define Omega  3.03008e-6   // solar angular velocity
#define rhelio 0.004633333  // solar radius in AU

// -----------------------------------------------------------------
// ------------------ tuned parameters -----------------------------
// -----------------------------------------------------------------


// -----------------------------------------------------------------
// ------------------  typedefs ------------------------------------
// -----------------------------------------------------------------


// In case of complex dependence this is the complete struct
// typedef struct {
//   float rr=0;    float rt=0;      float rp=0;
//   float tr=0;    float tt=0;      float tp=0;
//   float pr=0;    float pt=0;      float pp=0;
// } Tensor3D_t;
// typedef struct {
//   Tensor3D_t K;
//   float DKrr_dr=0; float DKtr_dt=0; float DKpr_dp=0;
//   float DKrt_dr=0; float DKtt_dt=0; float DKpt_dp=0;
//   float DKrp_dr=0; float DKtp_dt=0; float DKpp_dp=0;
// } DiffusionTensor_t;
//optimized - without unused variables and symmetric terms
typedef struct {
  float rr=0;    
  float tr=0;    float tt=0;      
  float pr=0;    float pt=0;      float pp=0;
} Tensor3D_t;
// __device__ Tensor3D_t initTensor3D_t(); // init struct to zero

typedef struct { 
  Tensor3D_t K;
  float DKrr_dr=0; float DKtr_dt=0; 
  float DKrt_dr=0; float DKtt_dt=0; 
  float DKrp_dr=0; float DKtp_dt=0; 
} DiffusionTensor_t;

typedef struct{
    float r         = 1.;       // heliocentric radial versor
    float th        = Half_Pi;  // heliocentric polar angle
    float phi       = 0.;       // heliocentric azimutal - longitudinal angle versor
} vect3D_t; // 3D vector

// typedef struct {  //total derivative
//   float dr;       // partial derivative on r
//   float dtheta;   // partial derivative on theta
//   float dphi;     // partial derivative on phi
// } derivative_t;



typedef struct{
    float r         = 1.;       // heliocentric radial versor
    float th        = Half_Pi;  // heliocentric polar angle
    float phi       = 0.;       // heliocentric azimutal - longitudinal angle versor
    float Ek        = 0.;       // kinetic energy (GeV/n) 
} qvect_t; // quadrivector


typedef struct{
    float T0  = 0.938; // rest mass in GeV/n
    float Z   = 1;     // Atomic number
    float A   = 1;     // mass number
} PartDescription_t; // particle type


typedef struct  {                                                   // data container for the particle information
    qvect_t part;
    PartDescription_t pt;       // particle type
    float alphapath = 0.;       // Montecarlo statistical Weight - exponent of c factor
    float prop_time = 0.;       // Total Propagation Time
} particle_t; // usage : 6*32bit 

typedef struct {                                                    // data container for output result for a single energy input
  unsigned long Nregistered;
  int           Nbins;
  float         LogBin0_lowEdge;  // lower boundary of first bin
  float         DeltaLogT;        // Bin amplitude in log scale
  float         *BoundaryDistribution;
} MonteCarloResult_t;

typedef struct {                                                  
  float Rts_nose = 100;             // Termination shock position 
  float Rhp_nose = 122;             // Heliopause position 
  float Rts_tail = 100;             // Termination shock position 
  float Rhp_tail = 122;             // Heliopause position 
} HeliosphereBoundRadius_t; 

// typedef struct{                                                     // properties related to heliosphere like dimension, coefficients
//   float Rmirror = 0.3 ;                                             // [AU] Internal heliosphere bounduary - mirror radius.
//   HeliosphereBoundRadius_t RadBoundary_effe;                        // boundaries in effective heliosphere
//   HeliosphereBoundRadius_t RadBoundary_real;                        // real boundaries heliosphere
//   short Nregions =  0 ;                                             // Number of Inner Heliosphere region
//   bool  IsHighActivityPeriod = false;                               // active the modification for high activity period
// } HeliosphereGeneralProperties_t;


// typedef struct{                                                     // properties related to heliosphere like dimension, coefficients
//   float V0;                                                         // [AU/s] Radial Solar wind speed on ecliptic
//   float k0_paral;                                                   // Diffusion parameter for parallel term of diffusion tensor
//   float k0_perp;                                                    // Diffusion parameter for perpendicular term of diffusion tensor  
//   float GaussVar;                                                   // Gaussian variation for Diffusion parameter
//   float g_low;                                                      // for evaluating Kpar, glow parameter
//   float rconst;                                                     // for evaluating Kpar, rconst parameter  
//   float TiltAngle;                                                  // Tilt angle of neutral sheet
//   float Asun;                                                       // normalization constant of HMF
//   float P0d;                                                        // Drift Suppression rigidity
//   float P0dNS;                                                      // NS Drift Suppression rigidity
//   int   Polarity;                                                   // HMF polarity
// } HeliosphereZoneProperties_t;


#define NMaxRegions 335    // about 25 year of simulations 
typedef struct{                                                     // properties related to heliosphere like dimension, coefficients
  float V0=0;                                                         // [AU/s] Radial Solar wind speed on ecliptic
  float k0_paral[2]={0};                                              // Diffusion parameter for parallel term of diffusion tensor [for HighActivity, for Low activity]
  float k0_perp[2]={0};                                               // Diffusion parameter for perpendicular term of diffusion tensor  [for HighActivity, for Low activity]
  float GaussVar[2]={0};                                              // Gaussian variation for Diffusion parameter[for HighActivity, for Low activity]
  float g_low=0;                                                      // for evaluating Kpar, glow parameter
  float rconst=0;                                                     // for evaluating Kpar, rconst parameter  
  float TiltAngle=0;                                                  // Tilt angle of neutral sheet
  float Asun=0;                                                       // normalization constant of HMF
  float P0d=0;                                                        // Drift Suppression rigidity
  float P0dNS=0;                                                      // NS Drift Suppression rigidity
  float plateau=0.;                                                   // Time dependent plateau in the high rigidity suppression
  signed char Polarity=0;                                             // HMF polarity
} HeliosphereZoneProperties_t;
typedef struct{                                                       // properties related to heliopause
  float V0=0;                                                         // [AU/s] Radial Solar wind speed on ecliptic
  float k0=0;                                                         // Diffusion parameter for parallel term of diffusion tensor
} HeliosheatProperties_t;
typedef struct{                                                     // properties related to heliosphere like dimension, coefficients
  float Rmirror = 0.3 ;                                             // [AU] Internal heliosphere bounduary - mirror radius.
  unsigned char Nregions =  0 ;                                     // Number of Inner Heliosphere region (15 inner region + 1 Heliosheat)
  HeliosphereBoundRadius_t RadBoundary_effe[NMaxRegions]={0};       // boundaries in effective heliosphere
  HeliosphereBoundRadius_t RadBoundary_real[NMaxRegions]={0};       // real boundaries heliosphere
  bool  IsHighActivityPeriod[NMaxRegions] = {false};                // active the modification for high activity period
//  HeliosphereZoneProperties_t prop_medium[NMaxRegions];             // PROPerties of the interplanetary MEDIUM - Heliospheric Parameters in each Heliospheric Zone
} SimulatedHeliosphere_t;




#endif

#ifndef MAINCU
// -----------------------------------------------------------------
// ------------  Device Constant Variables declaration -------------
// -----------------------------------------------------------------
extern __constant__ SimulatedHeliosphere_t  Heliosphere; // Heliosphere properties include Local Interplanetary medium parameters
extern __constant__ HeliosphereZoneProperties_t LIM[NMaxRegions];
extern __constant__ HeliosheatProperties_t      HS[NMaxRegions];        // heliosheat
#endif

__host__ __device__ float SmoothTransition(float , float , float , float , float );
__device__ float beta_(float , float );
__device__ float Rigidity(float ,PartDescription_t );
