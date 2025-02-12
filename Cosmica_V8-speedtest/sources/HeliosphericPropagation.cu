#include <stdio.h>
#include <curand_kernel.h>  // CUDA random number device library
#include <cuda_runtime.h>
#include "HeliosphericPropagation.cuh"
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"
#include "HeliosphereModel.cuh"
#include "SDECoeffs.cuh"
#include "GenComputation.cuh"
#include "Histogram.cuh"

// use template for the needs of unrolled max search in BlockMax
__global__ void HeliosphericProp(const int Npart_PerKernel, const float Min_dt, float Max_dt, const float TimeOut,
                                 QuasiParticle_t QuasiParts_out, const int *PeriodIndexes, const PartDescription_t pt,
                                 curandStatePhilox4_32_10_t *CudaState, float *RMaxs) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    curandStatePhilox4_32_10_t randState = CudaState[id];

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
    /* float* r      = &smem[threadIdx.x];
    float* th        = &smem[threadIdx.x + blockDim.x];
    float* phi       = &smem[threadIdx.x + 2*blockDim.x];
    float* R         = &smem[threadIdx.x + 3*blockDim.x];
    float* t_fly     = &smem[threadIdx.x + 4*blockDim.x];
    float* alphapath = &smem[threadIdx.x + 5*blockDim.x] ;
    float* zone      = &smem[threadIdx.x + 5*blockDim.x]; */

    // Execute only the thread filled with quasi-particle to propagate
    if (id < Npart_PerKernel) {
        // Copy the particle variables to shared memory for less latency
        smem[threadIdx.x] = QuasiParts_out.r[id];
        smem[threadIdx.x + blockDim.x] = QuasiParts_out.th[id];
        smem[threadIdx.x + 2 * blockDim.x] = QuasiParts_out.phi[id];
        smem[threadIdx.x + 3 * blockDim.x] = QuasiParts_out.R[id];
        smem[threadIdx.x + 4 * blockDim.x] = QuasiParts_out.t_fly[id];
        // smem[threadIdx.x + 5*blockDim.x] =  QuasiParts_out.alphapath[id];

        // Initialize the quasi particle position
#if TRIVIAL
      smem[threadIdx.x + 5*blockDim.x] = Zone(smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x]);
#else
        smem[threadIdx.x + 5 * blockDim.x] = RadialZone(PeriodIndexes[id], smem[threadIdx.x],
                                                        smem[threadIdx.x + blockDim.x],
                                                        smem[threadIdx.x + 2 * blockDim.x]);
#endif

        // Initialize the random state seed per thread
        //LocalState[id] = CudaState[id];

        // Stocasti propagation cycle until quasi-particle exit heliosphere or it reaches the fly timeout
        while (smem[threadIdx.x + 5 * blockDim.x] >= 0 && smem[threadIdx.x + 4 * blockDim.x] <= TimeOut) {
            // x,y,z used for SDE, w used for K0 random oscillation
            const auto [rand_x, rand_y, rand_z, rand_w] = curand_normal4(&randState);

            // Initi    alization of the propagation terms
            DiffusionTensor_t KSym;
            Tensor3D_t Ddif;
            vect3D_t AdvTerm;
            float en_loss;
            // float loss_term;
            float dt = Max_dt;

#if TRIVIAL
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

#else
            // Evaluate the convective-diffusive tensor and its decomposition
            KSym = DiffusionTensor_symmetric(PeriodIndexes[id], smem[threadIdx.x + 5 * blockDim.x], smem[threadIdx.x],
                                             smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2 * blockDim.x],
                                             smem[threadIdx.x + 3 * blockDim.x], pt, rand_w);

            int res = 0;
            Ddif = SquareRoot_DiffusionTerm(smem[threadIdx.x + 5 * blockDim.x], KSym, smem[threadIdx.x],
                                            smem[threadIdx.x + blockDim.x], &res);

            if (res > 0) {
                // SDE diffusion matrix is not positive definite; in this case propagation should be stopped and a new event generated
                // placing the energy below zero ensure that this event is ignored in the after-part of the analysis
                smem[threadIdx.x + 3 * blockDim.x] = -1;
                break; //exit the while cycle
            }


            // Evaluate advective-drift vector
            AdvTerm = AdvectiveTerm(PeriodIndexes[id], smem[threadIdx.x + 5 * blockDim.x], KSym, smem[threadIdx.x],
                                    smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2 * blockDim.x],
                                    smem[threadIdx.x + 3 * blockDim.x], pt);

            // Evaluate the energy loss term
            en_loss = EnergyLoss(PeriodIndexes[id], smem[threadIdx.x + 5 * blockDim.x], smem[threadIdx.x],
                                 smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2 * blockDim.x],
                                 smem[threadIdx.x + 3 * blockDim.x]);

            // Evaluate the loss term (Montecarlo statistical weight)
            // loss_term = LossTerm(PeriodIndexes[id], smem[threadIdx.x + 5*blockDim.x], smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x], smem[threadIdx.x + 3*blockDim.x], pt.T0);

            // evaluate time step
            dt = Max_dt;
            // time step is modified to ensure the diffusion approximation (i.e. diffusion step>>advective step)
            if (dt > Min_dt * (Ddif.rr * Ddif.rr) / (AdvTerm.r * AdvTerm.r))
                dt = max(Min_dt, Min_dt * (Ddif.rr * Ddif.rr) / (AdvTerm.r * AdvTerm.r));
            if (dt > Min_dt * (Ddif.tr + Ddif.tt) * (Ddif.tr + Ddif.tt) / (AdvTerm.th * AdvTerm.th))
                dt = max(Min_dt, Min_dt * (Ddif.tr + Ddif.tt) * (Ddif.tr + Ddif.tt) / (AdvTerm.th * AdvTerm.th));
#endif

            const float prev_r = smem[threadIdx.x];

            // Stochastic integration using the coefficients computed above and energy loss term
            smem[threadIdx.x] += AdvTerm.r * dt + rand_x * Ddif.rr * sqrtf(dt);

            // Reflect out the particle (use previous propagation step) if is closer to the sun than 0.3 AU
            if (smem[threadIdx.x] < Heliosphere.Rmirror) {
                smem[threadIdx.x] = prev_r;
            } else {
                smem[threadIdx.x + blockDim.x] += AdvTerm.th * dt + (rand_x * Ddif.tr + rand_y * Ddif.tt) *
                        sqrtf(dt);
                smem[threadIdx.x + 2 * blockDim.x] += AdvTerm.phi * dt + (
                    rand_x * Ddif.pr + rand_y * Ddif.pt + rand_z * Ddif.pp) * sqrtf(dt);
                smem[threadIdx.x + 3 * blockDim.x] += en_loss * dt;
                smem[threadIdx.x + 4 * blockDim.x] += dt;
                // smem[threadIdx.x + 5*blockDim.x] += loss_term*dt;
            }

            // Remap the polar coordinates inside their range (th in [0, Pi] & phi in [0, 2*Pi])
            smem[threadIdx.x + blockDim.x] = fabsf(smem[threadIdx.x + blockDim.x]);
            smem[threadIdx.x + blockDim.x] = fabsf(fmodf(
                2 * M_PI + safeSign(M_PI - smem[threadIdx.x + blockDim.x]) * smem[threadIdx.x + blockDim.x], M_PI));
            // --- reflecting latitudinal bounduary
            if (smem[threadIdx.x + blockDim.x] > thetaSouthlimit) {
                smem[threadIdx.x + blockDim.x] = 2 * thetaSouthlimit - smem[threadIdx.x + blockDim.x];
            } else if (smem[threadIdx.x + blockDim.x] < thetaNorthlimit) {
                smem[threadIdx.x + blockDim.x] = 2 * thetaNorthlimit - smem[threadIdx.x + blockDim.x];
            }

            smem[threadIdx.x + 2 * blockDim.x] = fmodf(smem[threadIdx.x + 2 * blockDim.x], 2 * M_PI);
            smem[threadIdx.x + 2 * blockDim.x] = fmodf(2 * M_PI + smem[threadIdx.x + 2 * blockDim.x], 2 * M_PI);

            // Check of the zone of heliosphere, heliosheat or interstellar medium where the quasi-particle is after a step
#if TRIVIAL
        smem[threadIdx.x + 5*blockDim.x] = Zone(smem[threadIdx.x], smem[threadIdx.x + blockDim.x], smem[threadIdx.x + 2*blockDim.x]);
#else
            smem[threadIdx.x + 5 * blockDim.x] = RadialZone(PeriodIndexes[id], smem[threadIdx.x],
                                                            smem[threadIdx.x + blockDim.x],
                                                            smem[threadIdx.x + 2 * blockDim.x]);
#endif
        }

        // Save peopagation exit values
        QuasiParts_out.r[id] = smem[threadIdx.x];
        QuasiParts_out.th[id] = smem[threadIdx.x + blockDim.x];
        QuasiParts_out.phi[id] = smem[threadIdx.x + 2 * blockDim.x];
        QuasiParts_out.R[id] = smem[threadIdx.x + 3 * blockDim.x];
        QuasiParts_out.t_fly[id] = smem[threadIdx.x + 4 * blockDim.x];
        // QuasiParts_out.alphapath[id] = smem[threadIdx.x + 5*blockDim.x];
    }

    // Find the maximum rigidity inside the block
    BlockMax(smem, RMaxs);
}
