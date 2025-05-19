#include <cstdio>
#include <curand_kernel.h>  // CUDA random number device library
#include <cuda_runtime.h>
#include "HeliosphericPropagation.cuh"
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"
#include "SDECoeffs.cuh"

/**
 * @brief Heliospheric propagation of cosmic rays in the heliosphere using a stochastic differential equation (SDE) approach
 * @param QuasiParts_out QuasiParticles output
 * @param indexes Indexes of the particles
 * @param params Simulation parameters
 * @param CudaState Random number generator state
 * @param RMaxs Maximum rigidity of the particles
 * @throws res If the diffusion tensor is not positive definite, the propagation is stopped and the energy is set to a negative value to signal the event
 */
__global__ void HeliosphericProp(ThreadQuasiParticles_t QuasiParts_out, const ThreadIndexes_t indexes,
                                 const SimulationParametrizations_t params, curandStatePhilox4_32_10_t *const CudaState,
                                 float *RMaxs) {
    const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= indexes.size) return;

    curandStatePhilox4_32_10_t randState = CudaState[id];

    auto qp = QuasiParts_out.get(id);
    auto index = indexes.get(id);
    index.update(qp);


    while (index.radial >= 0 && qp.t_fly <= Constants.Timeout) {
        const auto [rand_x, rand_y, rand_z, rand_w] = curand_normal4(&randState);

        auto KSym = DiffusionTensor_symmetric(index, qp, Constants.Isotopes[index.isotope], rand_w, params);

        int res = 0;
        const auto [rr, tr, tt, pr, pt, pp] = SquareRoot_DiffusionTerm(index, qp, KSym, &res);

        if (res > 0) {
            qp.R = -1;
            break;
        }

        const auto [adv_r, adv_th, adv_phi] = AdvectiveTerm(index, qp, KSym, Constants.Isotopes[index.isotope]);

        const float en_loss = EnergyLoss(index, qp);

        const float dt = fmaxf(Constants.Min_dt, fminf(
                                   fminf(Constants.Max_dt, Constants.Min_dt * (rr * rr) / (adv_r * adv_r)),
                                   Constants.Min_dt * (tr + tt) * (tr + tt) / (adv_th * adv_th)));

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
    QuasiParts_out.R[id] = qp.R;
    QuasiParts_out.t_fly[id] = qp.t_fly;

    CudaState[id] = randState;

    atomicMax(&RMaxs[index.instance(Constants.NIsotopes)], qp.R);
}
