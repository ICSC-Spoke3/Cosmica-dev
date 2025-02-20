#include <cstdio>
#include <curand_kernel.h>  // CUDA random number device library
#include <cuda_runtime.h>
#include "HeliosphericPropagation.cuh"
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"
#include "HeliosphereModel.cuh"
#include "SDECoeffs.cuh"
#include "GenComputation.cuh"
#include "Histogram.cuh"

__global__ void HeliosphericProp(const unsigned int Npart_PerKernel, const float Min_dt, float Max_dt,
                                 const float TimeOut,
                                 QuasiParticle_t QuasiParts_out, const ThreadIndexes indexes,
                                 const HeliosphereZoneProperties_t *__restrict__ LIM,
                                 curandStatePhilox4_32_10_t *const CudaState,
                                 float *RMaxs) {
    const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= Npart_PerKernel) return;

    extern __shared__ float smem[];

    curandStatePhilox4_32_10_t randState = CudaState[id];
    float r = QuasiParts_out.r[id];
    float th = QuasiParts_out.th[id];
    float phi = QuasiParts_out.phi[id];
    float R = QuasiParts_out.R[id];
    float t_fly = QuasiParts_out.t_fly[id];

    auto index = indexes.get(id);
    index.update(r, th, phi);


    while (index.radial >= 0 && t_fly <= TimeOut) {
        const auto [rand_x, rand_y, rand_z, rand_w] = curand_normal4(&randState);

        auto KSym = DiffusionTensor_symmetric(index, r, th, phi, R, Heliosphere.Isotopes[index.particle],
                                              rand_w, LIM);

        int res = 0;
        const auto [rr, tr, tt, pr, pt, pp] = SquareRoot_DiffusionTerm(index, KSym, r, th, &res);

        if (res > 0) {
            // SDE diffusion matrix is not positive definite; in this case propagation should be stopped and a new event generated
            // placing the energy below zero ensure that this event is ignored in the after-part of the analysis
            R = -1;
            break; //exit the while cycle
        }


        const auto [adv_r, adv_th, adv_phi] = AdvectiveTerm(index, KSym, r, th, phi, R, Heliosphere.Isotopes[index.particle],
                                                            LIM);

        const float en_loss = EnergyLoss(index, r, th, phi, R, LIM);

        const float dt = fmaxf(Min_dt, fminf(fminf(Max_dt,
                                                   Min_dt * (rr * rr) / (adv_r * adv_r)),
                                             Min_dt * (tr + tt) * (tr + tt) / (adv_th * adv_th)));


        if (const float update_r = r + adv_r * dt + rand_x * rr * sqrtf(dt); update_r >= Heliosphere.Rmirror) {
            r = update_r;
            th += adv_th * dt + (rand_x * tr + rand_y * tt) * sqrtf(dt);
            phi += adv_phi * dt + (rand_x * pr + rand_y * pt + rand_z * pp) * sqrtf(dt);
            R += en_loss * dt;
            t_fly += dt;
        }

        th = fabsf(th);
        th = fabsf(fmodf(2 * Pi + safeSign(Pi - th) * th, Pi));
        th = 2 * clamp(th, thetaNorthlimit, thetaSouthlimit) - th;

        phi = fmodf(phi, 2 * Pi);
        phi = fmodf(2 * Pi + phi, 2 * Pi);

        index.update(r, th, phi);
    }

    QuasiParts_out.r[id] = r;
    QuasiParts_out.th[id] = th;
    QuasiParts_out.phi[id] = phi;
    smem[threadIdx.x] = QuasiParts_out.R[id] = R;
    QuasiParts_out.t_fly[id] = t_fly;

    BlockMax(smem, RMaxs);
}
