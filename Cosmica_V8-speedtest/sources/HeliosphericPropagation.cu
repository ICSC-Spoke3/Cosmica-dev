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
                                 QuasiParticle_t QuasiParts_out, const int *PeriodIndexes,
                                 const PartDescription_t particle, curandStatePhilox4_32_10_t *const CudaState,
                                 float *RMaxs) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= Npart_PerKernel) return;

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

    // Execute only the thread filled with quasi-particle to propagate
    curandStatePhilox4_32_10_t randState = CudaState[id];
    // Copy the particle variables to shared memory for less latency
    float &r = smem[threadIdx.x] = QuasiParts_out.r[id];
    float &th = smem[threadIdx.x + blockDim.x] = QuasiParts_out.th[id];
    float &phi = smem[threadIdx.x + 2 * blockDim.x] = QuasiParts_out.phi[id];
    float &R = smem[threadIdx.x + 3 * blockDim.x] = QuasiParts_out.R[id];
    float &t_fly = smem[threadIdx.x + 4 * blockDim.x] = QuasiParts_out.t_fly[id];

    // Initialize the quasi particle position
    auto &rad_zone = reinterpret_cast<int32_t &>(smem[threadIdx.x + 5 * blockDim.x]) = RadialZone(
                         PeriodIndexes[id], r, th, phi);

    // Initialize the random state seed per thread
    //LocalState[id] = CudaState[id];

    // Stocasti propagation cycle until quasi-particle exit heliosphere or it reaches the fly timeout
    while (rad_zone >= 0 && t_fly <= TimeOut) {
        // x,y,z used for SDE, w used for K0 random oscillation
        const auto [rand_x, rand_y, rand_z, rand_w] = curand_normal4(&randState);

        // Initi    alization of the propagation terms
        DiffusionTensor_t KSym;
        Tensor3D_t Ddif;
        vect3D_t AdvTerm;
        float en_loss;
        float dt = Max_dt;

        // Evaluate the convective-diffusive tensor and its decomposition
        KSym = DiffusionTensor_symmetric(PeriodIndexes[id], rad_zone, r, th, phi, R, particle, rand_w);

        int res = 0;
        Ddif = SquareRoot_DiffusionTerm(rad_zone, KSym, r, th, &res);

        if (res > 0) {
            // SDE diffusion matrix is not positive definite; in this case propagation should be stopped and a new event generated
            // placing the energy below zero ensure that this event is ignored in the after-part of the analysis
            R = -1;
            break; //exit the while cycle
        }


        // Evaluate advective-drift vector
        AdvTerm = AdvectiveTerm(PeriodIndexes[id], rad_zone, KSym, r, th, phi, R, particle);

        // Evaluate the energy loss term
        en_loss = EnergyLoss(PeriodIndexes[id], rad_zone, r, th, phi, R);

        // Evaluate the loss term (Montecarlo statistical weight)
        // loss_term = LossTerm(PeriodIndexes[id], smem[threadIdx.x + 5*blockDim.x], r, th, smem[threadIdx.x + 2*blockDim.x], smem[threadIdx.x + 3*blockDim.x], pt.T0);

        // evaluate time step
        // time step is modified to ensure the diffusion approximation (i.e. diffusion step>>advective step)
        dt = fmaxf(Min_dt, fminf(fminf(Max_dt,
                                       Min_dt * (Ddif.rr * Ddif.rr) / (AdvTerm.r * AdvTerm.r)),
                                 Min_dt * (Ddif.tr + Ddif.tt) * (Ddif.tr + Ddif.tt) / (AdvTerm.th * AdvTerm.th)));


        // Stochastic integration using the coefficients computed above and energy loss term

        // Reflect out the particle (use previous propagation step) if is closer to the sun than 0.3 AU
        if (const float update_r = r + AdvTerm.r * dt + rand_x * Ddif.rr * sqrtf(dt); update_r >= Heliosphere.Rmirror) {
            r = update_r;
            th += AdvTerm.th * dt + (rand_x * Ddif.tr + rand_y * Ddif.tt) *
                    sqrtf(dt);
            phi += AdvTerm.phi * dt + (
                rand_x * Ddif.pr + rand_y * Ddif.pt + rand_z * Ddif.pp) * sqrtf(dt);
            R += en_loss * dt;
            t_fly += dt;
        }

        // Remap the polar coordinates inside their range (th in [0, Pi] & phi in [0, 2*Pi])
        th = fabsf(th);
        th = fabsf(fmodf(2 * M_PI + safeSign(M_PI - th) * th, M_PI));
        // --- reflecting latitudinal bounduary
        th = 2 * clamp(th, thetaNorthlimit, thetaSouthlimit) - th;

        phi = fmodf(phi, 2 * M_PI);
        phi = fmodf(2 * M_PI + phi, 2 * M_PI);

        // Check of the zone of heliosphere, heliosheat or interstellar medium where the quasi-particle is after a step
        rad_zone = RadialZone(PeriodIndexes[id], r, th, phi);
    }

    // Save peopagation exit values
    QuasiParts_out.r[id] = r;
    QuasiParts_out.th[id] = th;
    QuasiParts_out.phi[id] = phi;
    QuasiParts_out.R[id] = R;
    QuasiParts_out.t_fly[id] = t_fly;

    // Find the maximum rigidity inside the block
    BlockMax(smem, RMaxs);
}
