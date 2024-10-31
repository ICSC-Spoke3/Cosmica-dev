#include <stdio.h>
#include <math.h>
#include <curand.h>         // CUDA random number host library
#include <curand_kernel.h>  // CUDA random number device library
#include <cuda_runtime.h>
#include "HeliosphericPropagation.cuh"
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"
#include "HeliosphereModel.cuh"
#include "SDECoeffs.cuh"
#include "SolarWind.cuh"
#include "MagneticDrift.cuh"
#include "GenComputation.cuh"
#include "Histogram.cuh"

// use template for the needs of unrolled max search in BlockMax
__global__ void HeliosphericProp(int Npart_PerKernel, float Min_dt, float Max_dt, float TimeOut, struct QuasiParticle_t QuasiParts_out, int* PeriodIndexes, struct PartDescription_t pt, curandStatePhilox4_32_10_t* CudaState, float* RMaxs) {
  
  int id = threadIdx.x + blockDim.x * blockDim.x;

  printf("Entering HelProp, thread_id = %d\n", id);

  // Deine the external unique share memory array
  extern __shared__ float smem[];

  // __shared__ int Npart;
  // __shared__ float dt;
  // __shared__ float sqrtf(dt) = 0;
  // float MinValueTimeStep = Min_dt;
  // float MaxValueTimeStep = Max_dt;
  // int Npart = Npart_PerKernel;
  // struct PartDescription_t p_descr = pt;

  //__shared__ curandStatePhilox4_32_10_t LocalState[Npart_PerKernel];

  //__shared__ int ZoneNum[Npart_PerKernel];

  // subdivide the shared memory in the various variable arrays
  // CHECK TO NOT DOUBLE USE POINTERS WITH NOT NEEDED REGISTERS
  /*
  float* r      = &smem[threadIdx.x];
  float* th        = &smem[threadIdx.x + blockDim.x];
  float* phi       = &smem[threadIdx.x + 2*blockDim.x];
  float* R         = &smem[threadIdx.x + 3*blockDim.x];
  float* t_fly     = &smem[threadIdx.x + 4*blockDim.x];
  float* alphapath = &smem[threadIdx.x + 5*blockDim.x];
  float* zone      = &smem[threadIdx.x + 5*blockDim.x]; 
  float* KSym_rr = &smem[threadIdx.x + 6*blockDim.x];
  float* KSym_tr = &smem[threadIdx.x + 7*blockDim.x];
  float* KSym_tt = &smem[threadIdx.x + 8*blockDim.x];  
  float* KSym_pr = &smem[threadIdx.x + 9*blockDim.x];
  float* DKrr_dr = &smem[threadIdx.x + 10*blockDim.x];
  float* DKtr_dt = &smem[threadIdx.x + 11*blockDim.x];
  float* DKrt_dr = &smem[threadIdx.x + 12*blockDim.x];
  float* DKtt_dt = &smem[threadIdx.x + 13*blockDim.x];
  float* DKrp_dr = &smem[threadIdx.x + 14*blockDim.x];
  float* DKtp_dt = &smem[threadIdx.x + 15*blockDim.x];

  float* Ddif_rr = &smem[threadIdx.x + 16*blockDim.x];    
  float* Ddif_tr = &smem[threadIdx.x + 17*blockDim.x];
  float* Ddif_tt = &smem[threadIdx.x + 18*blockDim.x];      
  float* Ddif_pr = &smem[threadIdx.x + 19*blockDim.x];
  float* Ddif_pt = &smem[threadIdx.x + 20*blockDim.x];
  float* Ddif_pp = &smem[threadIdx.x + 21*blockDim.x];
  */

  // Execute only the thread filled with quasi-particle to propagate
  if (id<Npart_PerKernel) {

    // Copy the particle variables to shared memory for less latency
    smem[threadIdx.x]                =  QuasiParts_out.r[id];
    smem[threadIdx.x + blockDim.x]   =  QuasiParts_out.th[id];
    smem[threadIdx.x + 2*blockDim.x] =  QuasiParts_out.phi[id];
    smem[threadIdx.x + 3*blockDim.x] =  QuasiParts_out.R[id];
    smem[threadIdx.x + 4*blockDim.x] =  QuasiParts_out.t_fly[id];

    // Initialize the quasi particle position
    #if TRIVIAL
      smem[threadIdx.x + 5*blockDim.x] = Zone(smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x]);
    #else
      smem[threadIdx.x + 5*blockDim.x] = RadialZone(PeriodIndexes[id], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x]);
    #endif

    // Initialize the random state seed per thread
    //LocalState[id] = CudaState[id];

    // Stocasti propagation cycle until quasi-particle exit heliosphere or it reaches the fly timeout
    while(smem[threadIdx.x + 5*blockDim.x]>=0 && smem[threadIdx.x + 4*blockDim.x]<=TimeOut){

      float4 RandNum = curand_normal4(&CudaState[id]); // x,y,z used for SDE, w used for K0 random oscillation

      // Initialization of the propagation terms
      
      float en_loss;
      float dt = Max_dt;

      #if TRIVIAL
        struct DiffusionTensor_t KSym;
        struct Tensor_t Ddif;
        struct vect3D_t AdvTerm;

        // Evaluate the convective-diffusive tensor and its decomposition
        KSym = trivial_DiffusionTensor_symmetric(smem[threadIdx.x + 5*blockDim.x]); // Needed only to compute the two following terms
        Ddif = trivial_SquareRoot_DiffusionTensor(KSym);

        // Evaluate advective-drift vector
        AdvTerm = trivial_AdvectiveTerm(KSym);

        // Evaluate the energy loss term
        en_loss = fabsf(RandNum.x)*trivial_EnergyLoss();

        // Evaluate the loss term (Montecarlo statistical weight)
        // loss_term = 0;

        // Evaluate the time step of the SDE (dynamic or static time step? if it's dynamic would be also be individual for each thread?)
        dt = Max_dt/100;

        float prev_r = smem[threadIdx.x];

        // Stochastic integration using the coefficients computed above and energy loss term
        smem[threadIdx.x]                += AdvTerm.r*dt + RandNum.x*Ddif.rr*sqrtf(dt);

        // Reflect out the particle (use previous propagation step) if is closer to the sun than 0.3 AU
        if (smem[threadIdx.x]<Heliosphere.Rmirror) {
          smem[threadIdx.x] = prev_r;
        }
        
        else {
          smem[threadIdx.x + blockDim.x]   += AdvTerm.th*dt + (RandNum.x*Ddif.tr+RandNum.y*Ddif.tt)*sqrtf(dt);
          smem[threadIdx.x + 2*blockDim.x] += AdvTerm.phi*dt + (RandNum.x*Ddif.pr+RandNum.y*Ddif.pt+RandNum.z*Ddif.pp)*sqrtf(dt);
          smem[threadIdx.x + 3*blockDim.x] += en_loss*dt;
          smem[threadIdx.x + 4*blockDim.x] += dt;
        }

      #else
        {
          if(id==0) printf("Entering KSym zone\n");
          // Evaluate the convective-diffusive tensor and its decomposition (as single variables)
          struct DiffusionTensor_t KSym = DiffusionTensor_symmetric(PeriodIndexes[id], smem[threadIdx.x + 5*blockDim.x], smem[threadIdx.x],
                                          smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x], smem[threadIdx.x + 3*blockDim.x], pt, RandNum.w);

          smem[threadIdx.x + 6*blockDim.x]=KSym.rr;
          smem[threadIdx.x + 7*blockDim.x]=KSym.tr;
          smem[threadIdx.x + 8*blockDim.x]=KSym.tt;
          smem[threadIdx.x + 9*blockDim.x]=KSym.pr;
          smem[threadIdx.x + 10*blockDim.x]=KSym.pt;
          smem[threadIdx.x + 11*blockDim.x]=KSym.pp;
          smem[threadIdx.x + 12*blockDim.x]=KSym.DKrr_dr;
          smem[threadIdx.x + 13*blockDim.x]=KSym.DKtr_dt;
          smem[threadIdx.x + 14*blockDim.x]=KSym.DKrt_dr;
          smem[threadIdx.x + 15*blockDim.x]=KSym.DKtt_dt;
          smem[threadIdx.x + 16*blockDim.x]=KSym.DKrp_dr;
          smem[threadIdx.x + 17*blockDim.x]=KSym.DKtp_dt;

          // Evaluate the diffusive tensor (as single variables)
          struct Tensor3D_t Ddif = SquareRoot_DiffusionTerm(smem[threadIdx.x + 5*blockDim.x], smem[threadIdx.x + 6*blockDim.x], smem[threadIdx.x + 7*blockDim.x],
          smem[threadIdx.x + 8*blockDim.x], smem[threadIdx.x + 9*blockDim.x], smem[threadIdx.x + 10*blockDim.x], smem[threadIdx.x + 11*blockDim.x],
          smem[threadIdx.x], smem[threadIdx.x + blockDim.x]);
          
          smem[threadIdx.x + 18*blockDim.x]=Ddif.rr;    
          smem[threadIdx.x + 19*blockDim.x]=Ddif.tr;
          smem[threadIdx.x + 20*blockDim.x]=Ddif.tt;      
          smem[threadIdx.x + 21*blockDim.x]=Ddif.pr;
          smem[threadIdx.x + 22*blockDim.x]=Ddif.pt;
          smem[threadIdx.x + 23*blockDim.x]=Ddif.pp;
        }

        if(id==0) printf("Ddif.pp = %f", smem[threadIdx.x + 23*blockDim.x]);
        
        // Check if matrix square root is ok
        if ((isinf(smem[threadIdx.x + 16*blockDim.x]))||(isinf(smem[threadIdx.x + 17*blockDim.x]))||(isinf(smem[threadIdx.x + 18*blockDim.x]))||(isinf(smem[threadIdx.x + 19*blockDim.x]))||(isinf(smem[threadIdx.x + 20*blockDim.x]))||(isinf(smem[threadIdx.x + 21*blockDim.x]))
        ||(isnan(smem[threadIdx.x + 16*blockDim.x]))||(isnan(smem[threadIdx.x + 17*blockDim.x]))||(isnan(smem[threadIdx.x + 18*blockDim.x]))||(isnan(smem[threadIdx.x + 19*blockDim.x]))||(isnan(smem[threadIdx.x + 20*blockDim.x]))||(isnan(smem[threadIdx.x + 21*blockDim.x])) ){
          // SDE diffusion matrix is not positive definite; in this case propagation should be stopped and a new event generated
          // placing the energy below zero ensure that this event is ignored in the after-part of the analysis
          smem[threadIdx.x + 3*blockDim.x] = -1;   
          break; //exit the while cycle 
        }

        bool IsPolarRegion = (fabs(cos(smem[threadIdx.x + blockDim.x]))>CosPolarZone )? true:false;
        float dV_dth = DerivativeOfSolarWindSpeed_dtheta(PeriodIndexes[id], smem[threadIdx.x + 5*blockDim.x], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x]);
        float TiltPos_th = Pi/2.-LIM[PeriodIndexes[id] + int(smem[threadIdx.x + 5*blockDim.x])].TiltAngle;
        // float Asun = LIM[PeriodIndexes[id] + int(smem[threadIdx.x + 5*blockDim.x])].Asun; /* Magnetic Field Amplitude constant / aum^2*/
        // float TiltAngle = LIM[PeriodIndexes[id] + smem[threadIdx.x + 5*blockDim.x]].TiltAngle;
        // float TiltPos_r = smem[threadIdx.x];
        // float TiltPos_phi = smem[threadIdx.x + 2*blockDim.x];

        float Ka = eval_Ka(pt, smem[threadIdx.x + 3*blockDim.x]);
        
        float Vsw_PM89  = SolarWindSpeed(PeriodIndexes[id], smem[threadIdx.x + 5*blockDim.x], smem[threadIdx.x], TiltPos_th, smem[threadIdx.x + 2*blockDim.x]);
        
        float dthetans = fabs((GeV/(c*aum))*(2.*smem[threadIdx.x]*smem[threadIdx.x + 3*blockDim.x])/((LIM[PeriodIndexes[id] + int(smem[threadIdx.x + 5*blockDim.x])].Asun)*sqrt( 1+Gamma_Bfield(smem[threadIdx.x],TiltPos_th,Vsw_PM89)*Gamma_Bfield(smem[threadIdx.x],TiltPos_th,Vsw_PM89)+((IsPolarRegion)?delta_Bfield(smem[threadIdx.x],TiltPos_th)*delta_Bfield(smem[threadIdx.x],TiltPos_th):0))));
        float theta_mez = ((LIM[PeriodIndexes[id] + int(smem[threadIdx.x + 5*blockDim.x])].TiltAngle+dthetans)>Pi/2.)? Pi/2.-0.5*sin(Pi/2.): Pi/2.-0.5*sin(LIM[PeriodIndexes[id] + int(smem[threadIdx.x + 5*blockDim.x])].TiltAngle+dthetans); /* scaling parameter */
        float fth = eval_fth(smem[threadIdx.x + blockDim.x], theta_mez);
        float Dftheta_dtheta = eval_Dftheta_dtheta(smem[threadIdx.x + blockDim.x], theta_mez);
    
        float Vsw  = SolarWindSpeed(PeriodIndexes[id], smem[threadIdx.x + 5*blockDim.x], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x]); // solar wind evaluated
        float HighRigiSupp = eval_HighRigiSupp(PeriodIndexes[id], smem[threadIdx.x + 5*blockDim.x], smem[threadIdx.x + 3*blockDim.x]);

        vect3D_t v_drift = Drift_PM89(PeriodIndexes[id], smem[threadIdx.x + 5*blockDim.x], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x], smem[threadIdx.x + 3*blockDim.x], IsPolarRegion, LIM[PeriodIndexes[id] + int(smem[threadIdx.x + 5*blockDim.x])].Asun, Ka, fth, Dftheta_dtheta, Vsw, dV_dth, HighRigiSupp);

        // Evaluate advective-drift vector
        float AdvTerm_rad = AdvectiveTerm_radius(v_drift.r, PeriodIndexes[id], smem[threadIdx.x + 5*blockDim.x], smem[threadIdx.x + 6*blockDim.x], smem[threadIdx.x + 7*blockDim.x], smem[threadIdx.x + 10*blockDim.x], smem[threadIdx.x + 11*blockDim.x], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x], smem[threadIdx.x + 3*blockDim.x], pt);
        float AdvTerm_theta = 0;
        float AdvTerm_phi = 0;

        float prev_r = smem[threadIdx.x];

        // Stochastic integration using the coefficients computed above
        smem[threadIdx.x] += AdvTerm_rad*dt + RandNum.x*smem[threadIdx.x + 16*blockDim.x]*sqrtf(dt);

        // Reflect out the particle (use previous propagation step) if is closer to the sun than 0.3 AU
        if (smem[threadIdx.x]<Heliosphere.Rmirror) {
          smem[threadIdx.x] = prev_r;
        }

        else {
          // Evaluate advective-drift vector
          AdvTerm_theta = AdvectiveTerm_theta(v_drift.th, PeriodIndexes[id], smem[threadIdx.x + 5*blockDim.x], smem[threadIdx.x + 7*blockDim.x], smem[threadIdx.x + 8*blockDim.x], smem[threadIdx.x + 12*blockDim.x], smem[threadIdx.x + 13*blockDim.x], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x], smem[threadIdx.x + 3*blockDim.x], pt);

          // Stochastic integration using the coefficients computed above
          smem[threadIdx.x + blockDim.x]   += AdvTerm_theta*dt + (RandNum.x*smem[threadIdx.x + 17*blockDim.x]+RandNum.y*smem[threadIdx.x + 18*blockDim.x])*sqrtf(dt);

          // Evaluate advective-drift vector
          AdvTerm_phi = AdvectiveTerm_phi(v_drift.phi, PeriodIndexes[id], smem[threadIdx.x + 5*blockDim.x], smem[threadIdx.x + 9*blockDim.x], smem[threadIdx.x + 14*blockDim.x], smem[threadIdx.x + 15*blockDim.x], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x], smem[threadIdx.x + 3*blockDim.x], pt);

          // Stochastic integration using the coefficients computed above
          smem[threadIdx.x + 2*blockDim.x] += AdvTerm_phi*dt + (RandNum.x*smem[threadIdx.x + 19*blockDim.x]+RandNum.y*smem[threadIdx.x + 20*blockDim.x]+RandNum.z*smem[threadIdx.x + 21*blockDim.x])*sqrtf(dt);

          // Evaluate the energy loss term
          en_loss = EnergyLoss(PeriodIndexes[id], smem[threadIdx.x + 5*blockDim.x], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x], smem[threadIdx.x + 3*blockDim.x]);

          // Stochastic integration using energy loss term
          smem[threadIdx.x + 3*blockDim.x] += en_loss*dt;
          smem[threadIdx.x + 4*blockDim.x] += dt;

          if(id==0) printf("R = %f", smem[threadIdx.x + 3*blockDim.x]);

        }

        // evaluate time step
        dt = Max_dt;
        // time step is modified to ensure the diffusion approximation (i.e. diffusion step>>advective step)
        if (dt>Min_dt * (smem[threadIdx.x + 16*blockDim.x]*smem[threadIdx.x + 16*blockDim.x])/(AdvTerm_rad*AdvTerm_rad)) dt=max(Min_dt, Min_dt * (smem[threadIdx.x + 16*blockDim.x]*smem[threadIdx.x + 16*blockDim.x])/(AdvTerm_rad*AdvTerm_rad));
        if (dt>Min_dt * (smem[threadIdx.x + 17*blockDim.x]+smem[threadIdx.x + 18*blockDim.x])*(smem[threadIdx.x + 17*blockDim.x]+smem[threadIdx.x + 18*blockDim.x])/(AdvTerm_theta*AdvTerm_theta)) dt=max(Min_dt, Min_dt * (smem[threadIdx.x + 17*blockDim.x]+smem[threadIdx.x + 18*blockDim.x])*(smem[threadIdx.x + 17*blockDim.x]+smem[threadIdx.x + 18*blockDim.x])/(AdvTerm_theta*AdvTerm_theta));
      #endif

      // Remap the polar coordinates inside their range (th in [0, Pi] & phi in [0, 2*Pi])
      smem[threadIdx.x + blockDim.x] = fabsf(smem[threadIdx.x + blockDim.x]);
      smem[threadIdx.x + blockDim.x] = fabsf(fmodf(2*M_PI+sign(M_PI-smem[threadIdx.x + blockDim.x])*smem[threadIdx.x + blockDim.x], M_PI));
      // --- reflecting latitudinal bounduary
      if (smem[threadIdx.x + blockDim.x]>thetaSouthlimit) 
        {smem[threadIdx.x + blockDim.x] = 2*thetaSouthlimit-smem[threadIdx.x + blockDim.x];}
      else if (smem[threadIdx.x + blockDim.x]<(thetaNorthlimit))    
        {smem[threadIdx.x + blockDim.x] = 2*thetaNorthlimit-smem[threadIdx.x + blockDim.x];}

      smem[threadIdx.x + 2*blockDim.x] = fmodf(smem[threadIdx.x + 2*blockDim.x], 2*M_PI);
      smem[threadIdx.x + 2*blockDim.x] = fmodf(2*M_PI+smem[threadIdx.x + 2*blockDim.x], 2*M_PI);

      // Check of the zone of heliosphere, heliosheat or interstellar medium where the quasi-particle is after a step
      #if TRIVIAL
        smem[threadIdx.x + 5*blockDim.x] = Zone(smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x]);
      #else
        smem[threadIdx.x + 5*blockDim.x] = RadialZone(PeriodIndexes[id], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x]);
      #endif
    }

    // Save peopagation exit values
    QuasiParts_out.r[id]     = smem[threadIdx.x];                
    QuasiParts_out.th[id]    = smem[threadIdx.x + blockDim.x];   
    QuasiParts_out.phi[id]   = smem[threadIdx.x + 2*blockDim.x]; 
    QuasiParts_out.R[id]     = smem[threadIdx.x + 3*blockDim.x]; 
    QuasiParts_out.t_fly[id] = smem[threadIdx.x + 4*blockDim.x];
  }

  // Find the maximum rigidity inside the block
  BlockMax(smem, RMaxs);
}