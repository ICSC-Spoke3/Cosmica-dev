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
                                 ThreadQuasiParticles_t QuasiParts_out, const ThreadIndexes_t indexes,
                                 const HeliosphereZoneProperties_t *__restrict__ LIM,
                                 curandStatePhilox4_32_10_t *const CudaState,
                                 float *RMaxs) {
    const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= Npart_PerKernel) return;

    extern __shared__ float smem[];

    curandStatePhilox4_32_10_t randState = CudaState[id];

    auto qp = QuasiParts_out.get(id);
    auto index = indexes.get(id);
    index.update(qp);


    while (index.radial >= 0 && qp.t_fly <= TimeOut) {
        const auto [rand_x, rand_y, rand_z, rand_w] = curand_normal4(&randState);

        auto KSym = DiffusionTensor_symmetric(index, qp, Heliosphere.Isotopes[index.particle], rand_w, LIM);

        int res = 0;
        const auto [rr, tr, tt, pr, pt, pp] = SquareRoot_DiffusionTerm(index, qp, KSym, &res);

        if (res > 0) {
            // SDE diffusion matrix is not positive definite; in this case propagation should be stopped and a new event generated
            // placing the energy below zero ensure that this event is ignored in the after-part of the analysis
            qp.R = -1;
            break; //exit the while cycle
        }

        const auto [adv_r, adv_th, adv_phi] = AdvectiveTerm(index, qp, KSym, Heliosphere.Isotopes[index.particle], LIM);

        const float en_loss = EnergyLoss(index, qp, LIM);

        const float dt = fmaxf(Min_dt, fminf(fminf(Max_dt,
                                                   Min_dt * (rr * rr) / (adv_r * adv_r)),
                                             Min_dt * (tr + tt) * (tr + tt) / (adv_th * adv_th)));

        if (const float update_r = qp.r + adv_r * dt + rand_x * rr * sqrtf(dt); update_r >= r_mirror) {
            qp.r = update_r;
            qp.th += adv_th * dt + (rand_x * tr + rand_y * tt) * sqrtf(dt);
            qp.phi += adv_phi * dt + (rand_x * pr + rand_y * pt + rand_z * pp) * sqrtf(dt);
            qp.R += en_loss * dt;
            qp.t_fly += dt;
        }

        qp.normalize_angles();

        index.update(qp);
    }

    QuasiParts_out.r[id] = qp.r;
    QuasiParts_out.th[id] = qp.th;
    QuasiParts_out.phi[id] = qp.phi;
    smem[threadIdx.x] = QuasiParts_out.R[id] = qp.R;
    QuasiParts_out.t_fly[id] = qp.t_fly;

    BlockMax(smem, RMaxs);
}
