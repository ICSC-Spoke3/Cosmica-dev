#include <cuda_runtime.h>

#include "GPUManage.cuh"
#include "VariableStructure.cuh"
#include "GenComputation.cuh"

/**
 * @brief HandleError
 * @param err cudaError_t error
 * @param file the file where the error occurred
 * @param line the line where the error occurred
 */
void HandleError(const cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
        spdlog::critical("{} in {} at line {}", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief hash function to convert a string into a hash
 * @param str the string to be hashed
 * @return the hash of the string
 */
constexpr unsigned long hash(const std::string_view &str) {
    unsigned long hash = 0;
    for (const auto &e: str) hash = hash * 131 + e;
    return hash;
}

/**
 * @brief operator""_ to convert a string into a hash
 * @param str the string to be hashed
 * @param len the length of the string
 * @return the hash of the string
 */
consteval unsigned long operator""_(const char *str, const size_t len) {
    return hash(std::string_view(str, len));
}

/**
 * @brief Print the available GPUs
 * @return the number of GPUs
 */
int AvailableGPUs() {
    int NGPUs;
    HANDLE_ERROR(cudaGetDeviceCount(&NGPUs));
    return NGPUs;
}


/**
 * @brief Define the optimal Threads per Block value for a given GPU (name)
 * @param name the name of the GPU
 * @return the best number of threads per block
 */
int BestThreadsPerBlock(const std::string &name) {
    static const std::unordered_map<std::string, int> deviceTpB = {
        {"NVIDIA A30", 48},
        {"NVIDIA A40", 48},
        {"NVIDIA A100", 64} // TODO: temporary
    };

    int TpB = 32;  // Default value
    if (const auto it = deviceTpB.find(name); it != deviceTpB.end()) {
        TpB = it->second;
    } else {
        spdlog::warn("Optimal TpB value for device {} unknown, using {}", name, TpB);
    }

    spdlog::debug("Using {} Threads per Block for device {}", TpB, name);
    return TpB;
}

/**
 * @brief Round the number of particle to be simulated based on the GPU capability, returning a struct with threads
 * and blocks count together with shared memory bytes.
 * The calculations are done based on the cuda occupancy calculator output to maximize the usage of the GPUs.
 * @param NPart the number of particles
 * @param GPUprop the GPU properties
 * @return the launch parameters
 */
LaunchParam_t GetLaunchConfig(const unsigned NPart, cudaDeviceProp GPUprop) {
    auto launch_param = LaunchParam_t::from_TpB(BestThreadsPerBlock(GPUprop.name), NPart);

    if (launch_param.threads > static_cast<unsigned>(GPUprop.maxThreadsPerBlock)) {
        spdlog::critical("Error while configuring the Propagation Kernel");
        spdlog::critical("Too many threads per block: {} (max allowed {})", launch_param.threads,
                         GPUprop.maxThreadsPerBlock);
        exit(EXIT_FAILURE);
    }
    if (launch_param.blocks > static_cast<unsigned>(GPUprop.maxGridSize[0])) {
        spdlog::critical("Error while configuring the Propagation Kernel");
        spdlog::critical("Too many blocks per grid: {} (max allowed {})", launch_param.blocks, GPUprop.maxGridSize[0]);
        exit(EXIT_FAILURE);
    }

    spdlog::info("Propagation Kernel Configuration:");
    spdlog::info("* Number of particles      : {}", NPart);
    spdlog::info("* Number of blocks         : {}", launch_param.blocks);
    spdlog::info("* Number of threadsPerBlock: {}", launch_param.threads);

    return launch_param;
}

/**
 * @brief Retrieve the GPU properties (info) for the selected GPU and print a summary of useful infos for debugging,
 * with the verbose option.
 * @param N_GPU_count the number of GPU
 * @return the GPU properties
 */
cudaDeviceProp *DeviceInfo(const int N_GPU_count) {
    const auto infos = new cudaDeviceProp[N_GPU_count];

    for (int i = 0; i < N_GPU_count; i++)
        cudaGetDeviceProperties(&infos[i], i);

    spdlog::debug("GPU info:");

    for (int i = 0; i < N_GPU_count; i++) {
        spdlog::debug("* General Information for device {}:", i);
        spdlog::debug("  - Name:  {}", infos[i].name);
        spdlog::debug("  - Compute capability:  {}.{}", infos[i].major, infos[i].minor);
        spdlog::debug("  - Clock rate:  {}", infos[i].clockRate);
        spdlog::debug("  - Device copy overlap:  {}", infos[i].deviceOverlap ? "Enabled" : "Disabled");
        spdlog::debug("  - Kernel execution timeout :  {}", infos[i].kernelExecTimeoutEnabled ? "Enabled" : "Disabled");

        spdlog::debug("* Memory Information for device {}:", i);
        spdlog::debug("  - Total global mem:  {}", infos[i].totalGlobalMem);
        spdlog::debug("  - Total constant Mem:  {}", infos[i].totalConstMem);
        spdlog::debug("  - Max mem pitch:  {}", infos[i].memPitch);
        spdlog::debug("  - Texture Alignment:  {}", infos[i].textureAlignment);

        spdlog::debug("* MP Information for device {}:", i);
        spdlog::debug("  - Multiprocessor count:  {}", infos[i].multiProcessorCount);
        spdlog::debug("  - Shared mem per mp:  {}", infos[i].sharedMemPerBlock);
        spdlog::debug("  - Registers per mp:  {}", infos[i].regsPerBlock);
        spdlog::debug("  - Threads in warp:  {}", infos[i].warpSize);
        spdlog::debug("  - Max threads per block:  {}", infos[i].maxThreadsPerBlock);
        spdlog::debug("  - Max thread dimensions:  ({}, {}, {})", infos[i].maxThreadsDim[0], infos[i].maxThreadsDim[1],
                     infos[i].maxThreadsDim[2]);
        spdlog::debug("  - Max grid dimensions:  ({}, {}, {})", infos[i].maxGridSize[0], infos[i].maxGridSize[1],
                     infos[i].maxGridSize[2]);
    }

    return infos;
}
