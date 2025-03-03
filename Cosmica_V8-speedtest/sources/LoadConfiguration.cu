#include <cstdio>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions
#include "LoadConfiguration.cuh"
#include <memory>
#include "VariableStructure.cuh"

/**
 * @brief Allocate memory on the device and return a pointer to it
 * @return Pointer to the allocated memory
 * @throws HandleError if the memory allocation fails
 */
template<typename T> requires (!std::is_array_v<T>)
T *AllocateManaged() {
    T *ptr;
    HANDLE_ERROR(cudaMallocManaged(&ptr, sizeof(T)));
    return ptr;
}

/**
 * @brief Allocate memory on the device and return a pointer to it
 * @param size Size of the array to allocate
 * @return Pointer to the allocated memory
 * @throws HandleError if the memory allocation fails
 */
template<typename T> requires std::is_array_v<T>
std::remove_extent_t<T> *AllocateManaged(const size_t size) {
    using ET = std::remove_extent_t<T>;
    ET *ptr;
    HANDLE_ERROR(cudaMallocManaged(&ptr, size * sizeof(ET)));
    return ptr;
}

/**
 * @brief Allocate memory on the device and return a pointer to it; it's safe to use with std::unique_ptr
 * @return Pointer to the allocated memory
 */
template<typename T> requires (!std::is_array_v<T>)
auto AllocateManagedSafe() {
    return std::unique_ptr<T, decltype(&cudaFree)>(AllocateManaged<T>(), cudaFree);
}

/**
 * @brief Allocate memory on the device and return a pointer to it; it's safe to use with std::unique_ptr
 * @param size Size of the array to allocate
 * @return Pointer to the allocated memory
 */
template<typename T> requires std::is_array_v<T>
auto AllocateManagedSafe(const size_t size) {
    return std::unique_ptr<T, decltype(&cudaFree)>(AllocateManaged<T>(size), cudaFree);
}

/**
 * @brief Allocate memory on the device and return a pointer to it, initialized with the value v
 * @param v Value to initialize the memory with
 * @return Pointer to the allocated memory
 * @throws HandleError if the memory allocation fails
 */
template<typename T> requires (!std::is_array_v<T>)
T *AllocateManaged(const int v) {
    T *ptr = AllocateManaged<T>();
    HANDLE_ERROR(cudaMemset(ptr, v, sizeof(T)));
    return ptr;
}

/**
 * @brief Allocate memory on the device and return a pointer to it, initialized with the value v
 * @param size Size of the array to allocate
 * @param v Value to initialize the memory with
 * @return Pointer to the allocated memory
 * @throws HandleError if the memory allocation fails
 */
template<typename T> requires std::is_array_v<T>
std::remove_extent_t<T> *AllocateManaged(const size_t size, const int v) {
    using ET = std::remove_extent_t<T>;
    ET *ptr = AllocateManaged<T>(size);
    HANDLE_ERROR(cudaMemset(ptr, v, size * sizeof(ET)));
    return ptr;
}

/**
 * @brief Allocate memory on the device and return a pointer to it, initialized with the value v; it's safe to use with std::unique_ptr
 * @param v Value to initialize the memory with
 * @return Pointer to the allocated memory
 */
template<typename T> requires (!std::is_array_v<T>)
auto AllocateManagedSafe(const int v) {
    return std::unique_ptr<T, decltype(&cudaFree)>(AllocateManaged<T>(v), cudaFree);
}

/**
 * @brief Allocate memory on the device and return a pointer to it, initialized with the value v; it's safe to use with std::unique_ptr
 * @param size Size of the array to allocate
 * @param v Value to initialize the memory with
 * @return Pointer to the allocated memory
 */
template<typename T> requires std::is_array_v<T>
auto AllocateManagedSafe(const size_t size, const int v) {
    return std::unique_ptr<T, decltype(&cudaFree)>(AllocateManaged<T>(size, v), cudaFree);
}

/**
* @brief Allocate a ThreadQuasiParticles_t struct with the given number of particles
* @param NPart Number of particles to allocate
*/
ThreadQuasiParticles_t AllocateQuasiParticles(const unsigned NPart) {
    return {
        AllocateManaged<float[]>(NPart),
        AllocateManaged<float[]>(NPart),
        AllocateManaged<float[]>(NPart),
        AllocateManaged<float[]>(NPart),
        AllocateManaged<float[]>(NPart),
    };
}

/**
* @brief Copy the given value to the given constant symbol
* @param symbol Symbol to copy the value to
* @param src Value to copy
* @throws HandleError if the memory allocation fails
*/
template<typename T>
void CopyToConstant(const T &symbol, const T *src) {
    HANDLE_ERROR(cudaMemcpyToSymbol(symbol, src, sizeof(T)));
}

/**
* @brief Allocate a ThreadIndexes_t struct with the given number of particles
* @param NPart Number of particles to allocate
*/
ThreadIndexes_t AllocateIndex(const unsigned NPart) {
    return {
        NPart,
        AllocateManaged<unsigned[]>(NPart),
        AllocateManaged<unsigned[]>(NPart),
        AllocateManaged<unsigned[]>(NPart),
    };
}

/**
* @brief Allocate an InstanceHistograms struct with the given number of instances
* @param NRig Number of rigidities to allocate
* @param NInstances Number of instances to allocate
* @return Pointer to the allocated memory
*/
InstanceHistograms *AllocateResults(const unsigned NRig, const unsigned NInstances) {
    auto *res = AllocateManaged<InstanceHistograms[]>(NRig);
    for (unsigned i_rig = 0; i_rig < NRig; ++i_rig)
        res[i_rig] = AllocateManaged<MonteCarloResult_t[]>(NInstances);
    return res;
}

/**
* @brief Load the initial positions of the particles
* @param Npos Number of particles to load
* @param verbose Whether to print verbose output
* @return InitialPositions_t struct with the loaded values
*/
InitialPositions_t LoadInitPos(unsigned Npos, const bool verbose) {
    InitialPositions_t InitialPositions;

    InitialPositions.r = new float[Npos];
    InitialPositions.th = new float[Npos];
    InitialPositions.phi = new float[Npos];

    if (verbose) {
        printf("Default initial quasi particles configuration loaded\n");
    }

    return InitialPositions;
}

/**
* @brief Load the initial rigidities of the particles
* @param RBins Number of rigidities to load
* @param verbose Whether to print verbose output
* @return Array with the loaded values
*/
float *LoadInitRigidities(const int RBins, const bool verbose) {
    auto *rigidities = new float[RBins];

    for (int i = 0; i < RBins; i++) {
    }

    if (verbose) {
        printf("Default energies bins loaded\n");
    }

    return rigidities;
}

/**
* @brief Save the configuration of the simulation in a file
* @param filename Name of the file to save the configuration to
* @param NPart Number of particles
* @param Out_QuasiParts ThreadQuasiParticles_t struct with the particles
* @param RMax Maximum rigidity
* @param verbose Whether to print verbose output
* @throws EXIT_FAILURE if the file cannot be opened
*/
void SaveTxt_part(const char *filename, const int Npart, const ThreadQuasiParticles_t &Out_QuasiParts, const float RMax,
                  const bool verbose) {
    FILE *file = fopen(filename, "ab");
    if (file == nullptr) {
        printf("Error opening the file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fprintf(file, "r\t\t\t theta\t\t phi\t\t rigidity\t fly time\n");

    for (int i = 0; i < Npart; i++) {
        fprintf(file, "%f\t %f\t %f\t %f\t %f\n", Out_QuasiParts.r[i], Out_QuasiParts.th[i],
                Out_QuasiParts.phi[i], Out_QuasiParts.R[i], Out_QuasiParts.t_fly[i]);
    }

    fprintf(file, "max R\n%f\n\n", RMax);

    if (verbose) {
        printf("Propagation variables written successfully in file %s\n", filename);
    }

    fclose(file);
}

/**
* @brief Save the rigidities histogram in a file
* @param filename Name of the file to save the histogram to
* @param Bins Number of bins
* @param histo MonteCarloResult_t struct with the histogram
* @param verbose Whether to print verbose output
* @throws EXIT_FAILURE if the file cannot be opened
*/
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
