#ifndef MagneticDrift
#define MagneticDrift

#ifndef TDDS_P0d_Ini
 #define TDDS_P0d_Ini 0.500000
#endif
#ifndef TDDS_P0d_Fin
 #define TDDS_P0d_Fin 4.
#endif
#ifndef TDDS_P0d_CoT_asc
 #define TDDS_P0d_CoT_asc 73.
#endif
#ifndef TDDS_P0d_Smt_asc
 #define TDDS_P0d_Smt_asc 1.
#endif
#ifndef TDDS_P0d_CoT_des
 #define TDDS_P0d_CoT_des 65.
#endif
#ifndef TDDS_P0d_Smt_des
 #define TDDS_P0d_Smt_des 5.
#endif

#ifndef TDDS_P0dNS_Ini
 #define TDDS_P0dNS_Ini 0.500000
#endif
#ifndef TDDS_P0dNS_CoT_asc
 #define TDDS_P0dNS_CoT_asc 68.
#endif
#ifndef TDDS_P0dNS_Smt_asc
 #define TDDS_P0dNS_Smt_asc 1.
#endif
#ifndef TDDS_P0dNS_CoT_des
 #define TDDS_P0dNS_CoT_des 57.
#endif
#ifndef TDDS_P0dNS_Smt_des
 #define TDDS_P0dNS_Smt_des 5.
#endif

#ifndef SSNScalF
 #define SSNScalF 50.
#endif

#ifndef HRS_TransPoint_asc
 #define HRS_TransPoint_asc 35.
#endif
#ifndef HRS_smoothness_asc
 #define HRS_smoothness_asc 5.
#endif
#ifndef HRS_TransPoint_des
 #define HRS_TransPoint_des 40.
#endif
#ifndef HRS_smoothness_des
 #define HRS_smoothness_des 5.
#endif

#ifndef HighRigiSupp_smoothness
 #define HighRigiSupp_smoothness 0.3
#endif
#ifndef HighRigiSupp_TransPoint
 #define HighRigiSupp_TransPoint 9.5
#endif

// ------------------------------------------
__device__ float Gamma_Bfield(float , float, float );
__device__ float delta_Bfield(float , float);
__device__ vect3D_t Drift_PM89(unsigned char, signed char, float, float, float, float, struct PartDescription_t); 
/* * description: Evaluate the components of drift velocity according to Potgieter Mooral 1985 - See Burger&Hatttingh 1995 */
float EvalP0DriftSuppressionFactor(int , int ,float, float);
float EvalHighRigidityDriftSuppression_plateau(int , float);

#endif
