// ReSharper disable CppUnusedIncludeDirective
#define MAINCU

// .. standard C
#include <cstdio>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions
#include <cstdlib>         // Supplies malloc(), calloc(), and realloc()
#include <unistd.h>         // Supplies EXIT_FAILURE, EXIT_SUCCESS
#include <sys/types.h>      // Typedef shortcuts like uint32_t and uint64_t
#include <sys/time.h>       // supplies time()
#include <span>
#include <numeric>
#include <deque>
#include <ranges>

// .. multi-thread
#include <omp.h>

// math lib
#include <cmath>           // C math library
// .. CUDA specific
#include <curand_kernel.h>  // CUDA random number device library
#include <cuda_runtime.h>   // Device code management by providing implicit initialization, context management, and module management



// .. project specific
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <fkYAML/node.hpp>
#include "VariableStructure.cuh"

#ifndef UNIFIED_COMPILE
#include "LoadConfiguration.cuh"
#include "HeliosphericPropagation.cuh"
#include "GenComputation.cuh"
#include "HistoComputation.cuh"
#include "GPUManage.cuh"
#include "Histogram.cuh"

// .. old HelMod code
#include "HelModVariableStructure.cuh"
#include "IOConfiguration.cuh"
#include "DiffusionModel.cuh"
#endif
#include "HelModVariableStructure.cuh"

#include "EventSequence.hpp"


// Track the errors
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define ERR_NoOutputFile "ERROR: output file cannot be open, do you have writing permission?\n"

// Simulation iperparameters definition
#define WARPSIZE 32
#ifndef SetWarpPerBlock
#define SetWarpPerBlock -1                                // number of warp so be submitted -- modify this value to find the best performance



#endif
#define NPARTS 5000
#define NPOS 10
#define RBINS 100

// Debugging variables
#define VERBOSE 1
#define VERBOSE_2 1
#define VERBOSE_LOAD 0
#define INITSAVE 0
#define FINALSAVE 0

// Datas variables
#define MaxCharinFileName   90

// -----------------------------------------------------------------
// ------------  Device Constant Variables declaration -------------
// -----------------------------------------------------------------
__constant__ SimulationConstants_t Constants;


#ifdef UNIFIED_COMPILE
#include "sources/DiffusionModel.cu"
#include "sources/EventSequence.cpp"
#include "sources/GenComputation.cu"
#include "sources/GPUManage.cu"
#include "sources/HeliosphereModel.cu"
#include "sources/HeliosphericPropagation.cu"
#include "sources/IOConfiguration.cu"
#include "sources/HistoComputation.cu"
#include "sources/Histogram.cu"
#include "sources/LoadConfiguration.cu"
#include "sources/MagneticDrift.cu"
#include "sources/SDECoeffs.cu"
#include "sources/SolarWind.cu"
#endif


// Main Code
int main(int argc, char *argv[]) {
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    cli_options options = parse_cli_options(argc, argv);
    if (!options.log_file.empty()) {
        spdlog::drop("");
        set_default_logger(spdlog::basic_logger_mt("", options.log_file));
    }
    spdlog::set_level(options.log_level);

    spdlog::info("Verbosity Level: {}", to_string_view(options.log_level));
    if (options.use_stdin) spdlog::info("Using stdin");
    else spdlog::info("Input file: {}", options.input_file);
    if (options.use_stdout) spdlog::info("Using stdout");
    else spdlog::info("Output directory: {}", options.output_dir);
    if (options.legacy) spdlog::info("Using LEGACY formats");

    spdlog::info("Simulation started");

    const int NGPUs = AvailableGPUs();

    if (NGPUs < 1) {
        spdlog::critical("No CUDA capable devices were detected");
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp *GPUs_profile = DeviceInfo(NGPUs);
    omp_set_num_threads(NGPUs);

    spdlog::debug("CPU cores: {} (threads used {})", omp_get_num_procs(), omp_get_num_threads());

    SimConfiguration_t SimParameters;

    if (LoadConfigFile(options, SimParameters) != EXIT_SUCCESS) {
        spdlog::critical("Error while loading simulation parameters");
        exit(EXIT_FAILURE);
    }

    SimParameters.Npart = (SimParameters.Npart + NGPUs - 1) / NGPUs * NGPUs; // Round up to the closest multiple

    unsigned NRig = SimParameters.NT,
            NParams = SimParameters.simulation_parametrization.Nparams,
            NPositions = SimParameters.NInitialPositions,
            NIsotopes = SimParameters.simulation_constants.NIsotopes,
            NRep = SimParameters.Npart,
            NRepPerGPU = SimParameters.Npart / NGPUs;
    unsigned NInstances = NParams * NIsotopes,
            NPartsPerInstance = NPositions * NRep,
            NPartsPerInstancePerGPU = NPositions * NRepPerGPU;
    unsigned NParts = NRig * NInstances * NPartsPerInstance,
            NPartsPerGPU = NRig * NInstances * NPartsPerInstancePerGPU;
    spdlog::info("Simulation parameters loaded:");
    spdlog::info("# of rigidities: {}", NRig);
    spdlog::info("# of instances: {}", NInstances);
    spdlog::info("# particles per instance: {}", NPartsPerInstance);
    spdlog::info("# total particles: {}", NParts);

    std::string init_filename = SimParameters.output_file + "_prop_in.txt";
    std::string final_filename = SimParameters.output_file + "_prop_out.txt";
    std::string histo_filename = SimParameters.output_file + "_R_histo.txt";

    if (std::remove(init_filename.c_str()) != 0 || std::remove(final_filename.c_str()) != 0) {
        spdlog::warn("Error deleting old propagation files or they do not exist");
    } else {
        spdlog::info("Old propagation files deleted successfully");
    }

    if (std::remove(histo_filename.c_str()) != 0) {
        spdlog::warn("Error deleting old histogram files or it does not exist");
    } else {
        spdlog::info("Old histogram files deleted successfully");
    }

    EventSequence BENCHMARK{"Cosmica"};

    ThreadIndexes_t global_indexes = AllocateIndex(NParts);
    for (unsigned r = 0; r < NRig; ++r) {
        for (unsigned p = 0; p < NParams; ++p) {
            for (unsigned i = 0; i < NIsotopes; ++i) {
                for (unsigned o = 0; o < NPositions; ++o) {
                    for (unsigned x = 0; x < NRep; ++x) {
                        unsigned idx = x + NRep * (o + NPositions * (i + NIsotopes * (p + NParams * r)));
                        global_indexes.rig[idx] = r;
                        global_indexes.param[idx] = p;
                        global_indexes.isotope[idx] = i;
                        global_indexes.period[idx] = o;
                    }
                }
            }
        }
    }

    BENCHMARK.AddEvent("Index copied");

    ThreadQuasiParticles_t global_parts = AllocateQuasiParticles(NParts);
    for (unsigned iPart = 0; iPart < NParts; ++iPart) {
        global_parts.r[iPart] = SimParameters.InitialPositions.r[global_indexes.period[iPart]];
        global_parts.th[iPart] = SimParameters.InitialPositions.th[global_indexes.period[iPart]];
        global_parts.phi[iPart] = SimParameters.InitialPositions.phi[global_indexes.period[iPart]];
        if (SimParameters.UsingEnergy) {
            global_parts.R[iPart] = Rigidity(SimParameters.Tcentr[global_indexes.rig[iPart]],
                                             SimParameters.simulation_constants.Isotopes[global_indexes.isotope[
                                                 iPart]]);
        } else {
            global_parts.R[iPart] = SimParameters.Tcentr[global_indexes.rig[iPart]];
        }
        global_parts.t_fly[iPart] = 0;
    }

    auto global_res = SimParameters.Results = AllocateResults(NRig, NInstances);

    BENCHMARK.AddEvent("Particles allocated");

    spdlog::info("Global memory allocated");

    auto THREAD_SEQUENCE = BENCHMARK.StartSubsequence("Threads", true);
    vector<EventSequence::EventSequencePtr> THREAD_BENCHMARKS;
    for (int t = 0; t < NGPUs; ++t) {
        THREAD_BENCHMARKS.push_back(THREAD_SEQUENCE->StartSubsequence(fmt::format("GPU {}", t)));
    }

#pragma omp parallel
    {
        unsigned cpu_thread_id = omp_get_thread_num();
        unsigned gpu_id = cpu_thread_id % NGPUs;
        HANDLE_ERROR(cudaSetDevice(gpu_id));
        unsigned num_cpu_threads = omp_get_num_threads();

        spdlog::debug("CPU Thread {} (of {}) uses CUDA device {}", cpu_thread_id + 1, num_cpu_threads, gpu_id);

        cudaDeviceProp device_prop = GPUs_profile[gpu_id];
        auto [BLOCKS, THREADS] = GetLaunchConfig(NPartsPerGPU, device_prop);

        auto RandStates = AllocateManagedSafe<curandStatePhilox4_32_10_t[]>(NPartsPerGPU);
        unsigned long Rnd_seed = gpu_id + (SimParameters.RandomSeed == 0
                                               ? getpid() + time(nullptr)
                                               : SimParameters.RandomSeed);
        cudaDeviceSynchronize();
        init_rdmgenerator<<<BLOCKS, THREADS>>>(RandStates.get(), NPartsPerGPU, Rnd_seed);
        cudaDeviceSynchronize();

        THREAD_BENCHMARKS[cpu_thread_id]->AddEvent("Random State Initialized");

        spdlog::debug("Random Seeds Configured (seed {})", Rnd_seed);

        CopyToConstant(Constants, &SimParameters.simulation_constants);

        ThreadIndexes_t indexes = AllocateIndex(NPartsPerGPU);
        ThreadQuasiParticles_t QuasiParts = AllocateQuasiParticles(NPartsPerGPU);

        for (unsigned iPart = 0, iGlobal = gpu_id; iPart < NPartsPerGPU; ++iPart, iGlobal += NGPUs) {
            indexes.rig[iPart] = global_indexes.rig[iGlobal];
            indexes.param[iPart] = global_indexes.param[iGlobal];
            indexes.isotope[iPart] = global_indexes.isotope[iGlobal];
            indexes.period[iPart] = global_indexes.period[iGlobal];

            QuasiParts.r[iPart] = global_parts.r[iGlobal];
            QuasiParts.th[iPart] = global_parts.th[iGlobal];
            QuasiParts.phi[iPart] = global_parts.phi[iGlobal];
            QuasiParts.R[iPart] = global_parts.R[iGlobal];
            QuasiParts.t_fly[iPart] = global_parts.t_fly[iGlobal];
        }

        auto Maxs = AllocateManagedNested<float>(NRig, NInstances);
        auto Nfailed = AllocateManagedNested<unsigned>(NRig, NInstances, 0);

        auto Results = AllocateResults(NRig, NInstances);

        THREAD_BENCHMARKS[cpu_thread_id]->AddEvent("Index and particles copied");

        spdlog::debug("Local memory allocated [GPU: {}]", gpu_id);

        spdlog::info("Propagation start [GPU: {}]", gpu_id);

        cudaDeviceSynchronize();
        HeliosphericProp<<<BLOCKS, THREADS>>>(QuasiParts, indexes, SimParameters.simulation_parametrization,
                                              RandStates.get(), Maxs);
        cudaDeviceSynchronize();

        // std::ofstream out(SimParameters.output_file + "_t_fly.csv");
        // for (std::size_t i = 0; i < NPartsPerGPU; ++i) {
        //     out << QuasiParts.t_fly[i] << '\n';
        // }

        THREAD_BENCHMARKS[cpu_thread_id]->AddEvent("Propagation completed");

        spdlog::info("Propagation complete [GPU: {}]", gpu_id);

        // THREAD_BENCHMARKS[cpu_thread_id]->StartSubsequence("Histograms Allocation");
        for (unsigned iR = 0; iR < NRig; ++iR) {
            for (unsigned iI = 0; iI < NInstances; ++iI) {
                if (Maxs[iR][iI] < SimParameters.Tcentr[iR]) {
                    spdlog::error("The max exiting rigidity is smaller than initial one (Rig {}, Instance {})",
                                  iR, iI);
                    continue; //TODO: check if needed
                }

                float DeltaLogR = log10f(1.f + SimParameters.RelativeBinAmplitude);
                float LogBin0_lowEdge = log10f(SimParameters.Tcentr[iR]) - DeltaLogR / 2.f;
                float Bin0_lowEdge = powf(10, LogBin0_lowEdge);

                Results[iR][iI].Nbins = ceil(log10(Maxs[iR][iI] / Bin0_lowEdge) / DeltaLogR);
                Results[iR][iI].LogBin0_lowEdge = LogBin0_lowEdge;
                Results[iR][iI].DeltaLogR = DeltaLogR;

                Results[iR][iI].BoundaryDistribution = AllocateManaged<float[]>(Results[iR][iI].Nbins, 0);
                // THREAD_BENCHMARKS[cpu_thread_id]->AddEvent(fmt::format("Histogram {} Allocated", iI));
            }
        }
        THREAD_BENCHMARKS[cpu_thread_id]->AddEvent("Histograms allocated");

        cudaDeviceSynchronize();
        SimpleHistogram<<<BLOCKS, THREADS>>>(indexes, QuasiParts.R, Results, Nfailed);
        cudaDeviceSynchronize();

        for (unsigned iR = 0; iR < NRig; ++iR) {
            for (unsigned iI = 0; iI < NInstances; ++iI) {
                Results[iR][iI].Nregistered = NPartsPerInstancePerGPU - Nfailed[iR][iI];
#pragma omp critical
                {
                    if (global_res[iR][iI].Nbins == 0) {
                        global_res[iR][iI].Nregistered = Results[iR][iI].Nregistered;
                        global_res[iR][iI].Nbins = Results[iR][iI].Nbins;
                        global_res[iR][iI].LogBin0_lowEdge = Results[iR][iI].LogBin0_lowEdge;
                        global_res[iR][iI].DeltaLogR = Results[iR][iI].DeltaLogR;
                        global_res[iR][iI].BoundaryDistribution = AllocateManaged<float[]>(
                            global_res[iR][iI].Nbins);
                        for (unsigned b = 0; b < global_res[iR][iI].Nbins; ++b)
                            global_res[iR][iI].BoundaryDistribution[b] = Results[iR][iI].BoundaryDistribution[b];
                    } else {
                        global_res[iR][iI].Nregistered += Results[iR][iI].Nregistered;

                        if (global_res[iR][iI].Nbins < Results[iR][iI].Nbins) {
                            for (unsigned b = 0; b < global_res[iR][iI].Nbins; ++b)
                                Results[iR][iI].BoundaryDistribution[b] +=
                                        global_res[iR][iI].BoundaryDistribution[b];
                            HANDLE_ERROR(cudaFree(global_res[iR][iI].BoundaryDistribution));

                            global_res[iR][iI].Nbins = Results[iR][iI].Nbins;
                            global_res[iR][iI].BoundaryDistribution = AllocateManaged<float[]>(
                                global_res[iR][iI].Nbins, 0);
                        }

                        for (unsigned b = 0; b < Results[iR][iI].Nbins; ++b) {
                            global_res[iR][iI].BoundaryDistribution[b] += Results[iR][iI].BoundaryDistribution[b];
                        }
                    }
                }
            }
        }

        THREAD_BENCHMARKS[cpu_thread_id]->AddEvent("Histograms computed");

        spdlog::info("Histograms generated [GPU: {}]", gpu_id);

        for (unsigned iPart = 0, iGlobal = gpu_id; iPart < NPartsPerGPU; ++iPart, iGlobal += NGPUs) {
            global_parts.r[iGlobal] = QuasiParts.r[iPart];
            global_parts.th[iGlobal] = QuasiParts.th[iPart];
            global_parts.phi[iGlobal] = QuasiParts.phi[iPart];
            global_parts.R[iGlobal] = QuasiParts.R[iPart];
            global_parts.t_fly[iGlobal] = QuasiParts.t_fly[iPart];
        }

        THREAD_BENCHMARKS[cpu_thread_id]->AddEvent("Particles copied back");
    }

    THREAD_SEQUENCE->StopSubsequence();
    BENCHMARK.StopSubsequence();

    // Generate the YAML file name, following the old naming convention:
    if (StoreResults(options, SimParameters) != EXIT_SUCCESS) {
        spdlog::critical("Error while storing simulation results");
        exit(EXIT_FAILURE);
    }
    BENCHMARK.AddEvent("Output written");

    delete[] SimParameters.InitialPositions.r;
    delete[] SimParameters.InitialPositions.th;
    delete[] SimParameters.InitialPositions.phi;
    delete[] SimParameters.Tcentr;

    delete[] GPUs_profile;

    if (spdlog::get_level() == spdlog::level::trace) BENCHMARK.Log(spdlog::level::trace, 4);
    if (spdlog::get_level() == spdlog::level::debug) BENCHMARK.Log(spdlog::level::debug, 3);
    BENCHMARK.Log(spdlog::level::info, 2);

    spdlog::info("Simulation ended");

    return cudaDeviceReset();
}
