#include <stdio.h>
#include <math.h>
#include <curand.h>         // CUDA random number host library
#include <curand_kernel.h>  // CUDA random number device library
#include <cuda_runtime.h>
#include "HeliosphericPropagation.cuh"
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"
#include "HeliosphereLocation.cuh"
#include "HeliosphereModel.cuh"
#include "SDECoeffs.cuh"
#include "GenComputation.cuh"
#include "Histogram.cuh"

// PRopagation constants
#define MAX_DT 50.                                        // max allowed value of time step
#define MIN_DT 0.01                                       // min allowed value of time step
#define TIMEOUT 2.*10**7                                   // maximum quasi-particle flying time
             // std::numeric_limits<float>::infinity()

// use template for the needs of unrolled max search in BlockMax
__global__ void HeliosphericProp(int Npart_PerKernel, struct QuasiParticle_t QuasiParts_out, int* PeriodIndexes, struct PartDescription_t pt, curandStatePhilox4_32_10_t* CudaState, float* RMaxs) {

  int id = threadIdx.x + blockIdx.x * blockDim.x;

  // Deine the external unique share memory array
  extern __shared__ float smem[];

  // __shared__ int Npart;
  // __shared__ float dt;
  // __shared__ float sqrtf(dt) = 0;
  // float MinValueTimeStep = MIN_DT;
  // float MaxValueTimeStep = MAX_DT;
  // int Npart = Npart_PerKernel;
  // struct PartDescription_t p_descr = pt;

  //__shared__ curandStatePhilox4_32_10_t LocalState[Npart_PerKernel];

  //__shared__ int ZoneNum[Npart_PerKernel];

  // subdivide the shared memory in the various variable arrays
  // CHECK TO NOT DOUBLE USE POINTERS WITH NOT NEEDED REGISTERS
  /* float* r      = &smem[threadIdx.x];
  float* th        = &smem[threadIdx.x + blockDim.x];
  float* phi       = &smem[threadIdx.x + 2*blockDim.x];
  float* R         = &smem[threadIdx.x + 3*blockDim.x];
  float* t_fly     = &smem[threadIdx.x + 4*blockDim.x];
  float* alphapath = &smem[threadIdx.x + 5*blockDim.x] ;
  float* zone      = &smem[threadIdx.x + 6*blockDim.x];
  float* dt        = &smem[threadIdx.x + 7*blockDim.x];
  float* en_loss   = &smem[threadIdx.x + 8*blockDim.x];
  float loss_term  = &smem[threadIdx.x + 9*blockDim.x]; */

  // Execute only the thread filled with quasi-particle to propagate
  if (id<Npart_PerKernel) {

    // Copy the particle variables to shared memory for less latency
    smem[threadIdx.x]                =  QuasiParts_out.r[id];
    smem[threadIdx.x + blockDim.x]   =  QuasiParts_out.th[id];
    smem[threadIdx.x + 2*blockDim.x] =  QuasiParts_out.phi[id];
    smem[threadIdx.x + 3*blockDim.x] =  QuasiParts_out.R[id];
    smem[threadIdx.x + 4*blockDim.x] =  QuasiParts_out.t_fly[id];
    smem[threadIdx.x + 5*blockDim.x] =  QuasiParts_out.alphapath[id];

    // Initialize the quasi particle position
    #if TRIVIAL
      smem[threadIdx.x + 6*blockDim.x] = Zone(smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x]);
    #else
      smem[threadIdx.x + 6*blockDim.x] = RadialZone(PeriodIndexes[id], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x]);
    #endif

    smem[threadIdx.x + 7*blockDim.x] = MAX_DT;

    smem[threadIdx.x + 8*blockDim.x] = 0;
    smem[threadIdx.x + 9*blockDim.x] = 0;

    // Initialize the random state seed per thread
    //LocalState[id] = CudaState[id];
    unsigned int count = 0;

    // Stocasti propagation cycle until quasi-particle exit heliosphere or it reaches the fly timeout
    while(smem[threadIdx.x + 6*blockDim.x]>=0 && smem[threadIdx.x + 4*blockDim.x]<=TIMEOUT){

      float4 RandNum = curand_normal4(&CudaState[id]); // x,y,z used for SDE, w used for K0 random oscillation

      // Initialization of the propagation terms
      struct DiffusionTensor_t KSym;
      struct Tensor3D_t Ddif;
      struct vect3D_t AdvTerm;
      // float en_loss;
      // float loss_term;
      // float dt = MAX_DT;
      
      #if TRIVIAL
        // Evaluate the convective-diffusive tensor and its decomposition
        KSym = trivial_DiffusionTensor_symmetric(smem[threadIdx.x + 6*blockDim.x]); // Needed only to compute the two following terms
        Ddif = trivial_SquareRoot_DiffusionTensor(KSym);

        // Evaluate advective-drift vector
        AdvTerm = trivial_AdvectiveTerm(KSym);

        // Evaluate the energy loss term
        smem[threadIdx.x + 8*blockDim.x] = fabs(RandNum.x)*trivial_EnergyLoss();

        // Evaluate the loss term (Montecarlo statistical weight)
        smem[threadIdx.x + 9*blockDim.x] = 0;

        // Evaluate the time step of the SDE (dynamic or static time step? if it's dynamic would be also be individual for each thread?)
        smem[threadIdx.x + 7*blockDim.x] = MAX_DT/100;

      #else
        // Evaluate the convective-diffusive tensor and its decomposition
        KSym = DiffusionTensor_symmetric(PeriodIndexes[id], smem[threadIdx.x + 6*blockDim.x], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x], smem[threadIdx.x + 3*blockDim.x], pt, RandNum.w);

        int res = 0;
        Ddif = SquareRoot_DiffusionTerm(smem[threadIdx.x + 6*blockDim.x], KSym, smem[threadIdx.x], smem[threadIdx.x + blockDim.x], &res);

        if (res>0) {
          // SDE diffusion matrix is not positive definite; in this case propagation should be stopped and a new event generated
          // placing the energy below zero ensure that this event is ignored in the after-part of the analysis
          smem[threadIdx.x + 3*blockDim.x] = -1;   
          break; //exit the while cycle 
        }


        // Evaluate advective-drift vector
        AdvTerm = AdvectiveTerm(PeriodIndexes[id], smem[threadIdx.x + 6*blockDim.x], KSym, smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x], smem[threadIdx.x + 3*blockDim.x], pt);

        // Evaluate the energy loss term
        smem[threadIdx.x + 8*blockDim.x] = EnergyLoss(PeriodIndexes[id], smem[threadIdx.x + 6*blockDim.x], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x], smem[threadIdx.x + 3*blockDim.x], pt.T0);

        // Evaluate the loss term (Montecarlo statistical weight)
        // loss_term = LossTerm(PeriodIndexes[id], smem[threadIdx.x + 6*blockDim.x], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x], smem[threadIdx.x + 3*blockDim.x], pt.T0);
        smem[threadIdx.x + 9*blockDim.x] = LossTerm(PeriodIndexes[id], smem[threadIdx.x + 6*blockDim.x], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x], smem[threadIdx.x + 3*blockDim.x], pt.T0);

        // time step is modified to ensure the diffusion approximation (i.e. diffusion step>>advective step)
        smem[threadIdx.x + 7*blockDim.x] = MAX_DT;
        if (smem[threadIdx.x + 7*blockDim.x]>MIN_DT * (Ddif.rr*Ddif.rr)/(AdvTerm.r*AdvTerm.r))                     smem[threadIdx.x + 7*blockDim.x]=max(MIN_DT, MIN_DT * (Ddif.rr*Ddif.rr)                  /(AdvTerm.r*AdvTerm.r));
        if (smem[threadIdx.x + 7*blockDim.x]>MIN_DT * (Ddif.tr+Ddif.tt)*(Ddif.tr+Ddif.tt)/(AdvTerm.th*AdvTerm.th)) smem[threadIdx.x + 7*blockDim.x]=max(MIN_DT, MIN_DT * (Ddif.tr+Ddif.tt)*(Ddif.tr+Ddif.tt)/(AdvTerm.th*AdvTerm.th));
      #endif

      float prev_r = smem[threadIdx.x];

      // Stochastic integration using the coefficients computed above and energy loss term
      smem[threadIdx.x]                += AdvTerm.r*smem[threadIdx.x + 7*blockDim.x] + RandNum.x*Ddif.rr*sqrtf(smem[threadIdx.x + 7*blockDim.x]);

      // Reflect out the particle (use previous propagation step) if is closer to the sun than 0.3 AU
      if (smem[threadIdx.x]<Heliosphere.Rmirror) {
        smem[threadIdx.x] = prev_r;
      }
      
      else {
        smem[threadIdx.x + blockDim.x]   += AdvTerm.th*smem[threadIdx.x + 7*blockDim.x] + (RandNum.x*Ddif.tr+RandNum.y*Ddif.tt)*sqrtf(smem[threadIdx.x + 7*blockDim.x]);
        smem[threadIdx.x + 2*blockDim.x] += AdvTerm.phi*smem[threadIdx.x + 7*blockDim.x] + (RandNum.x*Ddif.pr+RandNum.y*Ddif.pt+RandNum.z*Ddif.pp)*sqrtf(smem[threadIdx.x + 7*blockDim.x]);
        smem[threadIdx.x + 3*blockDim.x] += smem[threadIdx.x + 8*blockDim.x]*smem[threadIdx.x + 7*blockDim.x];
        smem[threadIdx.x + 4*blockDim.x] += smem[threadIdx.x + 7*blockDim.x];
        smem[threadIdx.x + 5*blockDim.x] += smem[threadIdx.x + 9*blockDim.x]*smem[threadIdx.x + 7*blockDim.x];
      }

      // Remap the polar coordinates inside their range (th in [0, Pi] & phi in [0, 2*Pi])
      smem[threadIdx.x + blockDim.x] = fabs(smem[threadIdx.x + blockDim.x]);
      smem[threadIdx.x + blockDim.x] = fabs(fmodf(2*M_PI+sign(M_PI-smem[threadIdx.x + blockDim.x])*smem[threadIdx.x + blockDim.x], M_PI));
      // --- reflecting latitudinal bounduary
      if (smem[threadIdx.x + blockDim.x]>thetaSouthlimit) 
        {smem[threadIdx.x + blockDim.x] = 2*thetaSouthlimit-smem[threadIdx.x + blockDim.x];}
      else if (smem[threadIdx.x + blockDim.x]<(thetaNorthlimit))    
        {smem[threadIdx.x + blockDim.x] = 2*thetaNorthlimit-smem[threadIdx.x + blockDim.x];}

      smem[threadIdx.x + 2*blockDim.x] = fmodf(smem[threadIdx.x + 2*blockDim.x], 2*M_PI);
      smem[threadIdx.x + 2*blockDim.x] = fmodf(2*M_PI+smem[threadIdx.x + 2*blockDim.x], 2*M_PI);

      // Check of the zone of heliosphere, heliosheat or interstellar medium where the quasi-particle is after a step
      #if TRIVIAL
        smem[threadIdx.x + 6*blockDim.x] = Zone(smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x]);
      #else
        smem[threadIdx.x + 6*blockDim.x] = RadialZone(PeriodIndexes[id], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x]);
      #endif

      if (id==10) {
        printf("r = %f\t", smem[threadIdx.x]);
        printf("t_fly = %f\n", smem[threadIdx.x + 4*blockDim.x]);
      }
    }

    // Save peopagation exit values
    QuasiParts_out.r[id]     = smem[threadIdx.x];                
    QuasiParts_out.th[id]    = smem[threadIdx.x + blockDim.x];   
    QuasiParts_out.phi[id]   = smem[threadIdx.x + 2*blockDim.x]; 
    QuasiParts_out.R[id]     = smem[threadIdx.x + 3*blockDim.x]; 
    QuasiParts_out.t_fly[id] = smem[threadIdx.x + 4*blockDim.x];
    QuasiParts_out.alphapath[id] = smem[threadIdx.x + 5*blockDim.x];

    printf("Thread %d end propagation\n", id);
  }

  // Find the maximum rigidity inside the block
  BlockMax(smem, RMaxs);
  if (id==10) {
    printf("Thread 10 exit final HelProp\n");
  }
}