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
#include "spdlog/spdlog.h"
#include "headers/fkYAML.hpp"
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
#include "HelModLoadConfiguration.cuh"
#include "DiffusionModel.cuh"
#endif
#include "HelModVariableStructure.cuh"


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
#include "sources/GenComputation.cu"
#include "sources/GPUManage.cu"
#include "sources/HeliosphereModel.cu"
#include "sources/HeliosphericPropagation.cu"
#include "sources/HelModLoadConfiguration.cu"
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

    spdlog::set_level(spdlog::level::trace);
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

    if (LoadConfigFile(argc, argv, SimParameters, VERBOSE_LOAD) != EXIT_SUCCESS) {
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

    std::string init_filename = SimParameters.output_file_name + "_prop_in.txt";
    std::string final_filename = SimParameters.output_file_name + "_prop_out.txt";
    std::string histo_filename = SimParameters.output_file_name + "_R_histo.txt";

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

#define USE_RIGIDITY_QUEUE
#ifdef USE_RIGIDITY_QUEUE
    auto rig_indexes = std::views::iota(0u, SimParameters.NT);
    std::deque queue(rig_indexes.begin(), rig_indexes.end());
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

        cudaEvent_t start, MemorySet, Randomstep, stop;
        cudaEvent_t Cycle_start, Cycle_step00, Cycle_step0, Cycle_step1, Cycle_step2, InitialSave, FinalSave;
        if constexpr (VERBOSE) {
            HANDLE_ERROR(cudaEventCreate( &start ));
            HANDLE_ERROR(cudaEventCreate( &MemorySet ));
            HANDLE_ERROR(cudaEventCreate( &Randomstep ));
            HANDLE_ERROR(cudaEventCreate( &stop ));
            HANDLE_ERROR(cudaEventRecord( start, nullptr ));
        }

        auto RandStates = AllocateManagedSafe<curandStatePhilox4_32_10_t[]>(NParts);
        unsigned long Rnd_seed = SimParameters.RandomSeed == 0
                                     ? getpid() + time(nullptr) + gpu_id
                                     : SimParameters.RandomSeed;
        cudaDeviceSynchronize();
        init_rdmgenerator<<<BLOCKS, THREADS>>>(RandStates.get(), Rnd_seed);
        cudaDeviceSynchronize();

        if constexpr (VERBOSE) {
            HANDLE_ERROR(cudaEventRecord( Randomstep, nullptr ));
            HANDLE_ERROR(cudaEventSynchronize( Randomstep ));
            spdlog::debug("Random Seeds Configured (seed {})", Rnd_seed);
        }

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

        if constexpr (VERBOSE) {
            HANDLE_ERROR(cudaEventRecord( MemorySet, nullptr ));
            HANDLE_ERROR(cudaEventSynchronize( MemorySet ));
        }

#ifdef USE_RIGIDITY_QUEUE
        unsigned iR;
        while (test_and_pop(queue, iR)) {
#else
        for (unsigned int iR = gpu_id; iR < SimParameters.NT; iR += NGPUs) {
#endif
            spdlog::info("Simulation for rigidity {} [{}] started", SimParameters.Tcentr[iR], iR);

            if constexpr (VERBOSE) {
                HANDLE_ERROR(cudaEventCreate( &Cycle_start ));
                HANDLE_ERROR(cudaEventCreate( &Cycle_step00 ));
                HANDLE_ERROR(cudaEventCreate( &Cycle_step0 ));
                HANDLE_ERROR(cudaEventCreate( &Cycle_step1 ));
                HANDLE_ERROR(cudaEventCreate( &Cycle_step2 ));
                HANDLE_ERROR(cudaEventCreate( &InitialSave ));
                HANDLE_ERROR(cudaEventCreate( &FinalSave ));
                HANDLE_ERROR(cudaEventRecord( Cycle_start, nullptr ));
            }


            for (unsigned iPart = 0; iPart < NParts; ++iPart) {
                QuasiParts.r[iPart] = SimParameters.InitialPositions.r[indexes.period[iPart]];
                QuasiParts.th[iPart] = SimParameters.InitialPositions.th[indexes.period[iPart]];
                QuasiParts.phi[iPart] = SimParameters.InitialPositions.phi[indexes.period[iPart]];
                QuasiParts.R[iPart] = SimParameters.Tcentr[iR]; // TODO: dynamic rigidity based on isotope
                QuasiParts.t_fly[iPart] = 0;
            }


            auto Maxs = AllocateManagedSafe<float[]>(NInstances);
            auto Nfailed = AllocateManagedSafe<unsigned[]>(NInstances, 0);

            if constexpr (VERBOSE) {
                HANDLE_ERROR(cudaEventRecord( Cycle_step00, nullptr ));
                HANDLE_ERROR(cudaEventSynchronize( Cycle_step00 ));
            }

            if constexpr (INITSAVE) {
                SaveTxt_part(init_filename.c_str(), NParts, QuasiParts, Maxs[0]);
            }

            if constexpr (VERBOSE) {
                HANDLE_ERROR(cudaEventRecord( InitialSave, nullptr ));
                HANDLE_ERROR(cudaEventSynchronize( InitialSave ));
            }

            cudaDeviceSynchronize();
            HeliosphericProp<<<BLOCKS, THREADS>>>(QuasiParts, indexes, SimParameters.simulation_parametrization,
                                                  RandStates.get(), Maxs.get());
            cudaDeviceSynchronize();

            if constexpr (VERBOSE) {
                HANDLE_ERROR(cudaEventRecord( Cycle_step0, nullptr ));
                HANDLE_ERROR(cudaEventSynchronize( Cycle_step0 ));
            }

            if constexpr (VERBOSE) {
                HANDLE_ERROR(cudaEventRecord( Cycle_step1, nullptr ));
                HANDLE_ERROR(cudaEventSynchronize( Cycle_step1 ));
            }

            if constexpr (FINALSAVE) {
                SaveTxt_part(final_filename.c_str(), NParts, QuasiParts, Maxs[0]);
            }

            if constexpr (VERBOSE) {
                HANDLE_ERROR(cudaEventRecord( FinalSave, nullptr ));
                HANDLE_ERROR(cudaEventSynchronize( FinalSave ));
            }

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
            }

            cudaDeviceSynchronize();
            SimpleHistogram<<<BLOCKS, THREADS>>>(indexes, QuasiParts.R, Results[iR], Nfailed.get());
            cudaDeviceSynchronize();

            for (unsigned inst = 0; inst < NInstances; ++inst) {
                Results[iR][inst].Nregistered = NPartsPerInstance - Nfailed[inst];
                spdlog::debug("* Total Events.   : {}", NPartsPerInstance);
                spdlog::debug("* Recorded Events : {}", Nfailed[inst]);
                spdlog::debug("* Failed Events.  : {}", Results[iR][inst].Nregistered);
            }

            if constexpr (VERBOSE_2) {
                HANDLE_ERROR(cudaEventRecord( Cycle_step2, nullptr ));
                HANDLE_ERROR(cudaEventSynchronize( Cycle_step2 ));
                float Enl00, Enl0, Enl1, Enl2, EnlIn, EnlFin;
                HANDLE_ERROR(cudaEventElapsedTime( &Enl00,
                    Cycle_start, Cycle_step00 ));
                HANDLE_ERROR(cudaEventElapsedTime( &EnlIn,
                    Cycle_step00, InitialSave ));
                HANDLE_ERROR(cudaEventElapsedTime( &Enl0,
                    InitialSave, Cycle_step0 ));
                HANDLE_ERROR(cudaEventElapsedTime( &Enl1,
                    Cycle_step0, Cycle_step1 ));
                HANDLE_ERROR(cudaEventElapsedTime( &EnlFin,
                    Cycle_step1, FinalSave ));
                HANDLE_ERROR(cudaEventElapsedTime( &Enl2,
                    FinalSave, Cycle_step2 ));
                spdlog::debug("Performance for Rigidity {}:", iR);
                spdlog::debug("* Init               :  {:.3f} s", Enl00 * 1e-3);
                spdlog::debug("* Save initial state :  {:.3f} s", EnlIn * 1e-3);
                spdlog::debug("* Propagation phase  :  {:.3f} s", Enl0 * 1e-3);
                spdlog::debug("* Find Max           :  {:.3f} s", Enl1 * 1e-3);
                spdlog::debug("* Save final state   :  {:.3f} s", EnlFin * 1e-3);
                spdlog::debug("* Binning            :  {:.3f} s", Enl2 * 1e-3);
                HANDLE_ERROR(cudaEventDestroy( Cycle_start ));
                HANDLE_ERROR(cudaEventDestroy( Cycle_step00 ));
                HANDLE_ERROR(cudaEventDestroy( InitialSave ));
                HANDLE_ERROR(cudaEventDestroy( Cycle_step0 ));
                HANDLE_ERROR(cudaEventDestroy( Cycle_step1 ));
                HANDLE_ERROR(cudaEventDestroy( FinalSave ));
                HANDLE_ERROR(cudaEventDestroy( Cycle_step2 ));
            }
        }
        // end of the cycle on the rigidities

        if constexpr (VERBOSE) {
            HANDLE_ERROR(cudaEventRecord( stop, nullptr ));
            HANDLE_ERROR(cudaEventSynchronize( stop ));
        }
        // Execution Time
        if constexpr (VERBOSE) {
            float elapsedTime, firstStep, memset;
            HANDLE_ERROR(cudaEventElapsedTime( &memset, start, MemorySet ));
            HANDLE_ERROR(cudaEventElapsedTime( &firstStep, start, Randomstep ));
            HANDLE_ERROR(cudaEventElapsedTime( &elapsedTime, start, stop ));
            spdlog::debug("* Time to Set Memory : {:.3f} s", memset * 1e-3);
            spdlog::debug("* Time to create Rnd : {:.3f} s ({:.3f})", firstStep * 1e-3, (firstStep - memset) * 1e-3);
            spdlog::debug("* Time to execute    : {:.3f} s ({:.3f})", elapsedTime * 1e-3,
                          (elapsedTime - firstStep) * 1e-3);
        }

        if constexpr (VERBOSE) {
            HANDLE_ERROR(cudaEventDestroy( start ));
            HANDLE_ERROR(cudaEventDestroy( MemorySet ));
            HANDLE_ERROR(cudaEventDestroy( Randomstep ));
            HANDLE_ERROR(cudaEventDestroy( stop ));
        }

        spdlog::info("Simulation for rigidity {} [{}] ended", SimParameters.Tcentr[iR], iR);
    }
    // end of the multiple CPU thread pragma


    ////////////////////////////////////////////////////////////////
    //..... Exit results saving   ..................................
    ////////////////////////////////////////////////////////////////

    // Generate the YAML file name, following the old naming convention:
    // "<SimParameters.output_file_name>_matrix_<pid>.yaml"
    // char yamlFilename[MaxCharinFileName];
    // sprintf(yamlFilename, "%s_matrix_%lu.yaml", SimParameters.output_file_name, static_cast<unsigned long>(getpid()));
    std::string yamlFilename = fmt::format("{}_matrix_{}.yaml", SimParameters.output_file_name, getpid());

    try {
        write_results_yaml(yamlFilename, SimParameters);
        spdlog::info("Results saved to file: {}", yamlFilename);
    } catch (const std::exception &e) {
        std::cerr << "Error writing results to YAML file: " << e.what() << std::endl;
        return 1;
    }


    //  Save the summary histogram
    //  Free the dynamic memory

    // Save the rigidity histograms to txt file
    for (unsigned iR = 0; iR < SimParameters.NT; ++iR) {
        SaveTxt_histo(histo_filename.c_str(), Results[iR][0].Nbins, Results[iR][0]);
    }

    /* save results to file .dat */
    FILE *pFile_Matrix = nullptr;
    std::string datFilename = fmt::format("{}_matrix_{}.dat", SimParameters.output_file_name, getpid());

    spdlog::debug("Writing Output File: {}", datFilename);
    pFile_Matrix = fopen(datFilename.c_str(), "w");

    if (pFile_Matrix == nullptr) {
        spdlog::critical("Error, no output file");
        exit(EXIT_FAILURE);
    }

    fprintf(pFile_Matrix, "# COSMICA \n");
    if constexpr (VERBOSE) fprintf(pFile_Matrix, "# Number of Input energies;\n");
    fprintf(pFile_Matrix, "%d \n", SimParameters.NT);

    for (unsigned itemp = 0; itemp < SimParameters.NT; ++itemp) {
        if constexpr (VERBOSE) {
            fprintf(pFile_Matrix, "######  Bin %d \n", itemp);
            fprintf(pFile_Matrix,
                    "# Egen, Npart Gen., Npart Registered, Nbin output, log10(lower edge bin 0), Bin amplitude (in log scale)\n");
        }

        fprintf(pFile_Matrix, "%f %u %u %d %f %f \n", SimParameters.Tcentr[itemp],
                SimParameters.Npart,
                Results[itemp][0].Nregistered,
                Results[itemp][0].Nbins,
                Results[itemp][0].LogBin0_lowEdge,
                Results[itemp][0].DeltaLogR);
        if constexpr (VERBOSE) fprintf(pFile_Matrix, "# output distribution \n");

        for (int itNB = 0; itNB < Results[itemp][0].Nbins; itNB++) {
            fprintf(pFile_Matrix, "%e ", Results[itemp][0].BoundaryDistribution[itNB]);
        }


        fprintf(pFile_Matrix, "\n");
        fprintf(pFile_Matrix, "#\n"); // <--- dummy line to separate results
    }

    fflush(pFile_Matrix);
    fclose(pFile_Matrix);

    delete[] SimParameters.InitialPositions.r;
    delete[] SimParameters.InitialPositions.th;
    delete[] SimParameters.InitialPositions.phi;
    delete[] SimParameters.Tcentr;

    delete[] GPUs_profile;

    spdlog::info("Simulation ended");

    return cudaDeviceReset();
}
