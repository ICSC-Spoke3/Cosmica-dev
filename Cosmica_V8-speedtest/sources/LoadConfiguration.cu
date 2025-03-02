#include <cstdio>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions
#include "LoadConfiguration.cuh"

#include <memory>

#include "VariableStructure.cuh"

template<typename T> requires (!std::is_array_v<T>)
T *AllocateManaged() {
    T *ptr;
    HANDLE_ERROR(cudaMallocManaged(&ptr, sizeof(T)));
    return ptr;
}

template<typename T> requires std::is_array_v<T>
std::remove_extent_t<T> *AllocateManaged(const size_t size) {
    using ET = std::remove_extent_t<T>;
    ET *ptr;
    HANDLE_ERROR(cudaMallocManaged(&ptr, size * sizeof(ET)));
    return ptr;
}

template<typename T> requires (!std::is_array_v<T>)
auto AllocateManagedSafe() {
    return std::unique_ptr<T, decltype(&cudaFree)>(AllocateManaged<T>(), cudaFree);
}

template<typename T> requires std::is_array_v<T>
auto AllocateManagedSafe(const size_t size) {
    return std::unique_ptr<T, decltype(&cudaFree)>(AllocateManaged<T>(size), cudaFree);
}

template<typename T> requires (!std::is_array_v<T>)
T *AllocateManaged(const int v) {
    T *ptr = AllocateManaged<T>();
    HANDLE_ERROR(cudaMemset(ptr, v, sizeof(T)));
    return ptr;
}

template<typename T> requires std::is_array_v<T>
std::remove_extent_t<T> *AllocateManaged(const size_t size, const int v) {
    using ET = std::remove_extent_t<T>;
    ET *ptr = AllocateManaged<T>(size);
    HANDLE_ERROR(cudaMemset(ptr, v, size * sizeof(ET)));
    return ptr;
}

template<typename T> requires (!std::is_array_v<T>)
auto AllocateManagedSafe(const int v) {
    return std::unique_ptr<T, decltype(&cudaFree)>(AllocateManaged<T>(v), cudaFree);
}

template<typename T> requires std::is_array_v<T>
auto AllocateManagedSafe(const size_t size, const int v) {
    return std::unique_ptr<T, decltype(&cudaFree)>(AllocateManaged<T>(size, v), cudaFree);
}

ThreadQuasiParticles_t AllocateQuasiParticles(const unsigned NPart) {
    return {
        AllocateManaged<float[]>(NPart),
        AllocateManaged<float[]>(NPart),
        AllocateManaged<float[]>(NPart),
        AllocateManaged<float[]>(NPart),
        AllocateManaged<float[]>(NPart),
    };
}

template<typename T>
void CopyToConstant(const T &symbol, const T *src) {
    HANDLE_ERROR(cudaMemcpyToSymbol(symbol, src, sizeof(T)));
}

ThreadIndexes_t AllocateIndex(const unsigned NPart) {
    return {
        NPart,
        AllocateManaged<unsigned[]>(NPart),
        AllocateManaged<unsigned[]>(NPart),
        AllocateManaged<unsigned[]>(NPart),
    };
}

InstanceHistograms *AllocateResults(const unsigned NRig, const unsigned NInstances) {
    auto *res = AllocateManaged<InstanceHistograms[]>(NRig);
    for (unsigned i_rig = 0; i_rig < NRig; ++i_rig) res[i_rig] = AllocateManaged<MonteCarloResult_t[]>(NInstances);
    return res;
}

InitialPositions_t LoadInitPos(unsigned Npos, const bool verbose) {
    // Allocate the array memory
    InitialPositions_t InitialPositions;

    InitialPositions.r = new float[Npos];
    InitialPositions.th = new float[Npos];
    InitialPositions.phi = new float[Npos];

    // Set the InitialPositions arrays with default values

    if (verbose) {
        printf("Default initial quasi particles configuration loaded\n");
    }

    return InitialPositions;
}

float *LoadInitRigidities(const int RBins, const bool verbose) {
    // Allocate the array memory
    auto *rigidities = new float[RBins];

    // Set the InitialPositions arrays with default values
    for (int i = 0; i < RBins; i++) {
        rigidities[i] = sqrtf(static_cast<float>(i) + 1);
    }

    if (verbose) {
        printf("Default energies bins loaded\n");
    }

    return rigidities;
}

void SaveTxt_part(const char *filename, const int Npart, const ThreadQuasiParticles_t &Out_QuasiParts, const float RMax,
                  const bool verbose) {
    FILE *file = fopen(filename, "ab");
    if (file == nullptr) {
        printf("Error opening the file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fprintf(file, "r\t\t\t theta\t\t phi\t\t rigidity\t fly time\n"); // \t\t alphapath

    for (int i = 0; i < Npart; i++) {
        fprintf(file, "%f\t %f\t %f\t %f\t %f\n", Out_QuasiParts.r[i], Out_QuasiParts.th[i],
                Out_QuasiParts.phi[i], Out_QuasiParts.R[i], Out_QuasiParts.t_fly[i]); // Out_QuasiParts.alphapath[i]
    }

    fprintf(file, "max R\n%f\n\n", RMax);

    if (verbose) {
        printf("Propagation variables written successfully in file %s\n", filename);
    }

    fclose(file);
}

void SaveTxt_histo(const char *filename, const int Bins, const MonteCarloResult_t &histo, const bool verbose) {
    FILE *file = fopen(filename, "ab");
    if (file == nullptr) {
        printf("Error opening the file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fprintf(file, "Nbins = %d\t LogBin0_lowEdge = %f\t LogBin_Rmax = %f\t DeltaLogR = %f\n", histo.Nbins,
            histo.LogBin0_lowEdge, static_cast<float>(histo.Nbins) * histo.DeltaLogR, histo.DeltaLogR);
    fprintf(file, "Histogram counts:\n");

    for (int i = 0; i < Bins; i++) {
        fprintf(file, "%f\n", histo.BoundaryDistribution[i]);
    }

    if (verbose) {
        printf("Histo array written successfully\n");
    }

    fclose(file);
}
