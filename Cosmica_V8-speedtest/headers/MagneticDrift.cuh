#ifndef MagneticDrift
#define MagneticDrift

#ifndef TDDS_P0d_Ini
#define TDDS_P0d_Ini 0.500000f
#endif
#ifndef TDDS_P0d_Fin
#define TDDS_P0d_Fin 4.f
#endif
#ifndef TDDS_P0d_CoT_asc
#define TDDS_P0d_CoT_asc 73.f
#endif
#ifndef TDDS_P0d_Smt_asc
#define TDDS_P0d_Smt_asc 1.f
#endif
#ifndef TDDS_P0d_CoT_des
#define TDDS_P0d_CoT_des 65.f
#endif
#ifndef TDDS_P0d_Smt_des
#define TDDS_P0d_Smt_des 5.f
#endif

#ifndef TDDS_P0dNS_Ini
#define TDDS_P0dNS_Ini 0.500000f
#endif
#ifndef TDDS_P0dNS_CoT_asc
#define TDDS_P0dNS_CoT_asc 68.f
#endif
#ifndef TDDS_P0dNS_Smt_asc
#define TDDS_P0dNS_Smt_asc 1.f
#endif
#ifndef TDDS_P0dNS_CoT_des
#define TDDS_P0dNS_CoT_des 57.f
#endif
#ifndef TDDS_P0dNS_Smt_des
#define TDDS_P0dNS_Smt_des 5.f
#endif

#ifndef SSNScalF
#define SSNScalF 50.f
#endif

#ifndef HRS_TransPoint_asc
#define HRS_TransPoint_asc 35.f
#endif
#ifndef HRS_smoothness_asc
#define HRS_smoothness_asc 5.f
#endif
#ifndef HRS_TransPoint_des
#define HRS_TransPoint_des 40.f
#endif
#ifndef HRS_smoothness_des
#define HRS_smoothness_des 5.f
#endif

#ifndef HighRigiSupp_smoothness
#define HighRigiSupp_smoothness 0.3f
#endif
#ifndef HighRigiSupp_TransPoint
#define HighRigiSupp_TransPoint 9.5f
#endif

// ------------------------------------------
__device__ float Gamma_Bfield(float, float, float);

__device__ float delta_Bfield(float, float);

__device__ vect3D_t Drift_PM89(const Index_t &, const QuasiParticle_t&, PartDescription_t, const HeliosphereZoneProperties_t *LIM);

/* * description: Evaluate the components of drift velocity according to Potgieter Mooral 1985 - See Burger&Hatttingh 1995 */
float EvalP0DriftSuppressionFactor(int, int, float, float);

float EvalHighRigidityDriftSuppression_plateau(int, float);

#endif
