#include "Histogram.cuh"
#include <VariableStructure.cuh>

/**
 * @brief Kernel to compute the histogram of the boundary distribution of the particles
 * @param indexes The indexes of the particles
 * @param R The distance of the particles
 * @param histograms The histograms of the instances
 * @param failed The number of particles that failed to be added to the histogram
 */
__global__ void SimpleHistogram(const ThreadIndexes_t indexes, const float *R, InstanceHistograms histograms,
                                unsigned *failed) {
    const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= indexes.size) return;

    const auto index = indexes.get(id);
    const auto inst = index.instance(Constants.NIsotopes);
    const auto &hist = histograms[inst];

    if (log10f(R[id]) > hist.LogBin0_lowEdge) {
        const int DestBin = static_cast<int>(floorf((log10f(R[id]) - hist.LogBin0_lowEdge) / hist.DeltaLogR));
        atomicAdd(&hist.BoundaryDistribution[DestBin], 1);
    } else {
        atomicAdd(&failed[inst], 1);
    }
}
