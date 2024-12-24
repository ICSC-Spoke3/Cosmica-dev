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
 #define TDDS_P0d_CoT_des 63.
#endif
#ifndef TDDS_P0d_Smt_des
 #define TDDS_P0d_Smt_des 10.
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



// ------------------------------------------
__device__ float Gamma_Bfield(float , float, float );
__device__ float delta_Bfield(float , float);
__device__ vect3D_t Drift_PM89(unsigned short, signed short , qvect_t ,PartDescription_t ); 
/* * description: Evaluate the components of drift velocity according to Potgieter Mooral 1985 - See Burger&Hatttingh 1995 */
float EvalP0DriftSuppressionFactor(int , int ,float, float);


#endif
