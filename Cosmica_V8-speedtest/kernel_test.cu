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

bool test_and_pop(std::deque<unsigned> &queue, unsigned &val) {
    bool ret;
#pragma omp critical
    {
        if ((ret = !queue.empty())) {
            val = queue.front();
            queue.pop_front();
        }
    }
    return ret;
}

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

    unsigned NParams = SimParameters.simulation_parametrization.Nparams, NPositions = SimParameters.NInitialPositions,
            NIsotopes = SimParameters.simulation_constants.NIsotopes, NRep = SimParameters.Npart;
    unsigned NInstances = NParams * NIsotopes, NPartsPerInstance = NPositions * NRep;
    unsigned NParts = NInstances * NPartsPerInstance;
    spdlog::info("Simulation parameters loaded:");
    spdlog::info("# of instances: {}", NInstances);
    spdlog::info("# particles per instance: {}", NPartsPerInstance);
    spdlog::info("# total particles: {}", NParts);

    auto Results = SimParameters.Results = AllocateResults(SimParameters.NT, NParts);

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

    EventSequence BENCHMARK{"Cosmica", true};
    vector<EventSequence::EventSequencePtr> THREAD_BENCHMARKS;
    for (int t = 0; t < NGPUs; ++t) {
        THREAD_BENCHMARKS.push_back(BENCHMARK.StartSubsequence(fmt::format("GPU {}", t)));
    }

#define USE_RIGIDITY_QUEUE
#ifdef USE_RIGIDITY_QUEUE
    auto rig_indexes = std::views::iota(0u, SimParameters.NT);
    std::deque<unsigned> queue{rig_indexes.begin(), rig_indexes.end()};
    // std::deque<unsigned> queue;
    // for (unsigned i = 0; i < SimParameters.NT; ++i) queue.push_back(i);
#endif

#pragma omp parallel
    {
        unsigned cpu_thread_id = omp_get_thread_num();
        unsigned gpu_id = cpu_thread_id % NGPUs;
        HANDLE_ERROR(cudaSetDevice(gpu_id));
        unsigned num_cpu_threads = omp_get_num_threads();

        spdlog::debug("CPU Thread {} (of {}) uses CUDA device {}", cpu_thread_id + 1, num_cpu_threads, gpu_id);

        cudaDeviceProp device_prop = GPUs_profile[gpu_id];
        auto [BLOCKS, THREADS] = GetLaunchConfig(NParts, device_prop);

        auto RandStates = AllocateManagedSafe<curandStatePhilox4_32_10_t[]>(NParts);
        unsigned long Rnd_seed = SimParameters.RandomSeed == 0
                                     ? getpid() + time(nullptr) + gpu_id
                                     : SimParameters.RandomSeed;
        cudaDeviceSynchronize();
        init_rdmgenerator<<<BLOCKS, THREADS>>>(RandStates.get(), Rnd_seed);
        cudaDeviceSynchronize();

        THREAD_BENCHMARKS[cpu_thread_id]->AddEvent("Random State Initialized");

        spdlog::debug("Random Seeds Configured (seed {})", Rnd_seed);

        CopyToConstant(Constants, &SimParameters.simulation_constants);

        ThreadQuasiParticles_t QuasiParts = AllocateQuasiParticles(NParts);

        ThreadIndexes_t indexes = AllocateIndex(NParts);
        for (unsigned p = 0; p < NParams; ++p) {
            for (unsigned i = 0; i < NIsotopes; ++i) {
                for (unsigned o = 0; o < NPositions; ++o) {
                    for (unsigned x = 0; x < NRep; ++x) {
                        unsigned idx = x + NRep * (o + NPositions * (i + NIsotopes * p));
                        indexes.param[idx] = p;
                        indexes.isotope[idx] = i;
                        indexes.period[idx] = o;
                    }
                }
            }
        }

        THREAD_BENCHMARKS[cpu_thread_id]->AddEvent("Shared Data Allocated");

#ifdef USE_RIGIDITY_QUEUE
        unsigned iR;
        while (test_and_pop(queue, iR)) {
#else
        for (unsigned int iR = gpu_id; iR < SimParameters.NT; iR += NGPUs) {
#endif
            spdlog::info("Simulation for rigidity {} [{}] started", SimParameters.Tcentr[iR], iR);

            THREAD_BENCHMARKS[cpu_thread_id]->StartSubsequence(
                fmt::format("Rigidity {:.3e} [{:02}]", SimParameters.Tcentr[iR], iR));

            for (unsigned iPart = 0; iPart < NParts; ++iPart) {
                QuasiParts.r[iPart] = SimParameters.InitialPositions.r[indexes.period[iPart]];
                QuasiParts.th[iPart] = SimParameters.InitialPositions.th[indexes.period[iPart]];
                QuasiParts.phi[iPart] = SimParameters.InitialPositions.phi[indexes.period[iPart]];
                QuasiParts.R[iPart] = SimParameters.Tcentr[iR]; // TODO: dynamic rigidity based on isotope
                QuasiParts.t_fly[iPart] = 0;
            }


            auto Maxs = AllocateManagedSafe<float[]>(NInstances);
            auto Nfailed = AllocateManagedSafe<unsigned[]>(NInstances, 0);

            THREAD_BENCHMARKS[cpu_thread_id]->AddEvent("Particles Data Allocated");

            if constexpr (INITSAVE && options.legacy) {
                SaveTxt_part(init_filename.c_str(), NParts, QuasiParts, Maxs[0]);
                THREAD_BENCHMARKS[cpu_thread_id]->AddEvent("Initial State Stored");
            }


            cudaDeviceSynchronize();
            HeliosphericProp<<<BLOCKS, THREADS>>>(QuasiParts, indexes, SimParameters.simulation_parametrization,
                                                  RandStates.get(), Maxs.get());
            cudaDeviceSynchronize();

            THREAD_BENCHMARKS[cpu_thread_id]->AddEvent("Propagation Completed");

            if constexpr (FINALSAVE && options.legacy) {
                SaveTxt_part(final_filename.c_str(), NParts, QuasiParts, Maxs[0]);
                THREAD_BENCHMARKS[cpu_thread_id]->AddEvent("Final State Stored");
            }


            THREAD_BENCHMARKS[cpu_thread_id]->StartSubsequence("Histograms Allocation");
            for (unsigned inst = 0; inst < NInstances; ++inst) {
                spdlog::debug("Results for Instance {} (Rigidity {}):", inst, iR);
                spdlog::debug("* R_min: {}, R_max: {}", SimParameters.Tcentr[iR], Maxs[0]);

                if (Maxs[inst] < SimParameters.Tcentr[iR]) {
                    spdlog::error("The max exiting rigidity is smaller than initial one (Instance {})", inst);
                    continue; //TODO: check if needed
                }

                float DeltaLogR = log10f(1.f + SimParameters.RelativeBinAmplitude);
                float LogBin0_lowEdge = log10f(SimParameters.Tcentr[iR]) - DeltaLogR / 2.f;
                float Bin0_lowEdge = powf(10, LogBin0_lowEdge);

                Results[iR][inst].Nbins = ceil(log10(Maxs[inst] / Bin0_lowEdge) / DeltaLogR);
                Results[iR][inst].LogBin0_lowEdge = LogBin0_lowEdge;
                Results[iR][inst].DeltaLogR = DeltaLogR;

                Results[iR][inst].BoundaryDistribution = AllocateManaged<float[]>(Results[iR][inst].Nbins, 0);
                THREAD_BENCHMARKS[cpu_thread_id]->AddEvent(fmt::format("Histogram {} Allocated", inst));
            }
            THREAD_BENCHMARKS[cpu_thread_id]->StopSubsequence();

            cudaDeviceSynchronize();
            SimpleHistogram<<<BLOCKS, THREADS>>>(indexes, QuasiParts.R, Results[iR], Nfailed.get());
            cudaDeviceSynchronize();

            for (unsigned inst = 0; inst < NInstances; ++inst) {
                Results[iR][inst].Nregistered = NPartsPerInstance - Nfailed[inst];
                spdlog::debug("* Total Events.   : {}", NPartsPerInstance);
                spdlog::debug("* Recorded Events : {}", Nfailed[inst]);
                spdlog::debug("* Failed Events.  : {}", Results[iR][inst].Nregistered);
            }

            THREAD_BENCHMARKS[cpu_thread_id]->AddEvent("Histograms Generated");

            THREAD_BENCHMARKS[cpu_thread_id]->StopSubsequence();

            spdlog::info("Simulation for rigidity {} [{}] ended", SimParameters.Tcentr[iR], iR);
        }
        // end of the cycle on the rigidities
    }
    // end of the multiple CPU thread pragma


    ////////////////////////////////////////////////////////////////
    //..... Exit results saving   ..................................
    ////////////////////////////////////////////////////////////////

    // Generate the YAML file name, following the old naming convention:
    if (StoreResults(options, SimParameters) != EXIT_SUCCESS) {
        spdlog::critical("Error while storing simulation results");
        exit(EXIT_FAILURE);
    }

    delete[] SimParameters.InitialPositions.r;
    delete[] SimParameters.InitialPositions.th;
    delete[] SimParameters.InitialPositions.phi;
    delete[] SimParameters.Tcentr;

    delete[] GPUs_profile;


    if (spdlog::get_level() == spdlog::level::trace) BENCHMARK.Log(spdlog::level::trace, 3);
    if (spdlog::get_level() == spdlog::level::debug) BENCHMARK.Log(spdlog::level::debug, 2);
    BENCHMARK.Log(spdlog::level::info, 1);

    spdlog::info("Simulation ended");

    return cudaDeviceReset();
}
