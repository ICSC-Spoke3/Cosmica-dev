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

    spdlog::set_level(spdlog::level::debug);
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
    printf("NInstances: %u, PPI: %u, Parts: %u\n", NInstances, NPartsPerInstance, NParts);
    spdlog::info("Simulation parameters loaded:");
    spdlog::info("# of instances: {}", NInstances);
    spdlog::info("# particles per instance: {}", NPartsPerInstance);
    spdlog::info("# total particles: {}", NParts);


    ////////////////////////////////////////////////////////////////
    //..... Rescale Heliosphere to an effective one  ...............
    ////////////////////////////////////////////////////////////////

    // Allocation of the output results for all the rigidities
    auto *OldResults = new MonteCarloResult_t[SimParameters.NT];
    auto Results = SimParameters.Results = AllocateResults(SimParameters.NT, NParts);

    // .. Results saving files

    // Initial and final results files
    char file_trivial[8] = {};

    char init_filename[20];
    sprintf(init_filename, "%sprop_in.txt", file_trivial);
    char final_filename[ReadingStringLenght];
    sprintf(final_filename, "%s_%sprop_out.txt", SimParameters.output_file_name, file_trivial);
    // Clean previous files
    if (remove(init_filename) != 0 || remove(final_filename) != 0)
        printf(
            "Error deleting the old propagation files or it does not exist\n");
    else printf("Old propagation files deleted successfully\n");

    // Initial and final results files
    char histo_filename[20];
    sprintf(histo_filename, "%sR_histo.txt", file_trivial);
    // Clean previous files
    if (remove(histo_filename) != 0) printf("Error deleting the old histogram files or it does not exist\n");
    else printf("Old histogram files deleted successfully\n");
    printf("-- \n\n");


    ////////////////////////////////////////////////////////////////
    //..... Simulations initialization   ...........................
    ////////////////////////////////////////////////////////////////

    //  Start CPU pragma menaging its own portion of the data
    //  Optimation of the number of particles, threads, blocks and
    //  shared memory with respect the GPU hardware

    // start cpu threads
#define USE_RIGIDITY_QUEUE
#ifdef USE_RIGIDITY_QUEUE
    auto rig_indexes = std::views::iota(0u, SimParameters.NT);
    std::deque queue(rig_indexes.begin(), rig_indexes.end());
#endif

#pragma omp parallel
    {
        // Grep the CPU and GPU id and set them
        unsigned int cpu_thread_id = omp_get_thread_num(); // identificativo del CPU-thread
        unsigned int gpu_id = cpu_thread_id % NGPUs;
        // seleziona la id della GPU da usare. "% num_gpus" allows more CPU threads than GPU devices
        HANDLE_ERROR(cudaSetDevice(gpu_id)); // seleziona la GPU
        unsigned int num_cpu_threads = omp_get_num_threads(); // numero totale di CPU-thread allocated

        if constexpr (VERBOSE) {
            printf("----- Individual CPU infos -----\n");
            printf("-- CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);
            printf("-- \n\n");
        }


        // Retrive information from the set GPU
        cudaDeviceProp device_prop = GPUs_profile[gpu_id];

        // Rounding the number of particle and calculating threads, blocks and share memory to acheive the maximum usage of the GPUs
        auto [BLOCKS, THREADS] = GetLaunchConfig(NParts, device_prop);

        ////////////////////////////////////////////////////////////////
        //..... capture the start time of GPU part .....................
        //      This part is for debug and performances tests
        ////////////////////////////////////////////////////////////////
        cudaEvent_t start, MemorySet, Randomstep, stop;
        cudaEvent_t Cycle_start, Cycle_step00, Cycle_step0, Cycle_step1, Cycle_step2, InitialSave, FinalSave;
        if constexpr (VERBOSE) {
            HANDLE_ERROR(cudaEventCreate( &start ));
            HANDLE_ERROR(cudaEventCreate( &MemorySet ));
            HANDLE_ERROR(cudaEventCreate( &Randomstep ));
            HANDLE_ERROR(cudaEventCreate( &stop ));
            HANDLE_ERROR(cudaEventRecord( start, nullptr ));
        }
        ////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////
        //..... GPU execution initialization   .........................
        ////////////////////////////////////////////////////////////////

        //  Set pseudo random number generator seeds
        //  Device memory allocation and threads starting positions


        // .. Initialize random generator
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

        // Allocate the initial variables and allocate on device
        ThreadQuasiParticles_t QuasiParts = AllocateQuasiParticles(NParts);

        // Period along which CR are integrated and the corresponding period indecies
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


        // Recording the setting memory execution time
        if constexpr (VERBOSE) {
            HANDLE_ERROR(cudaEventRecord( MemorySet, nullptr ));
            HANDLE_ERROR(cudaEventSynchronize( MemorySet ));
        }

        ////////////////////////////////////////////////////////////////
        //..... GPU perticle propagation   .............................
        ////////////////////////////////////////////////////////////////

        //  Initialization of the cycle on rigidities bins (for all the positions)
        //  Launch of the GPU propagation kernel (computing diffusion
        //  coefficients and solving stochastic differential equations)
        //  Build the exit energy histogram

        // Cycle on rigidity bins distributing their execution between the active CPU threads
#ifdef USE_RIGIDITY_QUEUE
        unsigned iR;
        while (test_and_pop(queue, iR)) {
#else
        for (unsigned int iR = gpu_id; iR < SimParameters.NT; iR += NGPUs) {
#endif
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

            spdlog::debug("Simulation for rigidity {} [{}]", SimParameters.Tcentr[iR], iR);

            // Initialize the particle starting rigidities
            for (unsigned iPart = 0; iPart < NParts; ++iPart) {
                QuasiParts.r[iPart] = SimParameters.InitialPositions.r[indexes.period[iPart]];
                QuasiParts.th[iPart] = SimParameters.InitialPositions.th[indexes.period[iPart]];
                QuasiParts.phi[iPart] = SimParameters.InitialPositions.phi[indexes.period[iPart]];
                QuasiParts.R[iPart] = SimParameters.Tcentr[iR];
                QuasiParts.t_fly[iPart] = 0;
            }


            // Allocate the array for the partial rigidities maxima and final maximum
            auto Maxs = AllocateManagedSafe<float[]>(NInstances);
            auto Nfailed = AllocateManagedSafe<unsigned[]>(NInstances, 0);

            if constexpr (VERBOSE) {
                HANDLE_ERROR(cudaEventRecord( Cycle_step00, nullptr ));
                HANDLE_ERROR(cudaEventSynchronize( Cycle_step00 ));
            }

            // Saving the initial particles parameters into a txt file for debugging
            if constexpr (INITSAVE) {
                SaveTxt_part(init_filename, NParts, QuasiParts, Maxs[0], VERBOSE_2);
            }

            if constexpr (VERBOSE) {
                HANDLE_ERROR(cudaEventRecord( InitialSave, nullptr ));
                HANDLE_ERROR(cudaEventSynchronize( InitialSave ));
            }


            // Heliosphere propagation kernel
            // and local max rigidity search inside the block
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

            // if constexpr (FINALSAVE) {
            //     // host final states for specific energy
            //     ThreadQuasiParticles_t host_final_QuasiParts = AllocateQuasiParticles(NParts);
            //
            //     SaveTxt_part(final_filename, NParts, host_final_QuasiParts, Maxs[0], VERBOSE_2);
            //
            //     // Free the host particle variable for the energy on which the cycle is running
            //     free(host_final_QuasiParts.r);
            //     free(host_final_QuasiParts.th);
            //     free(host_final_QuasiParts.phi);
            //     free(host_final_QuasiParts.R);
            //     free(host_final_QuasiParts.t_fly);
            //     // free(host_final_QuasiParts.alphapath);
            // }

            if constexpr (VERBOSE) {
                HANDLE_ERROR(cudaEventRecord( FinalSave, nullptr ));
                HANDLE_ERROR(cudaEventSynchronize( FinalSave ));
            }

            for (unsigned inst = 0; inst < NInstances; ++inst) {
                if constexpr (VERBOSE_2)
                    fprintf(stdout, "\n--- RMin = %.3f Rmax = %.3f \n", SimParameters.Tcentr[iR],
                            Maxs[0]);

                if (Maxs[inst] < SimParameters.Tcentr[iR]) {
                    printf("PROBLEMA: the max exiting rigidity is smaller than initial one\n");
                    continue; //TODO: check if needed
                }

                float DeltaLogR = log10f(1.f + SimParameters.RelativeBinAmplitude);
                float LogBin0_lowEdge = log10f(SimParameters.Tcentr[iR]) - DeltaLogR / 2.f;
                float Bin0_lowEdge = powf(10, LogBin0_lowEdge); // first LowEdge Bin

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
                if constexpr (VERBOSE_2) {
                    fprintf(stdout, "-- Eventi computati : %u \n", NPartsPerInstance);
                    fprintf(stdout, "-- Eventi falliti   : %u \n", Nfailed[inst]);
                    fprintf(stdout, "-- Eventi registrati: %u \n", Results[iR][inst].Nregistered);
                }
            }

            OldResults[iR] = Results[iR][0];

            // ANNOTATION THE ONLY MEMCOPY NEEDED FROM DEVICE TO HOST ARE THE FINAL RESULTS (ALIAS THE ENERGY FINAL HISTOGRAM AND PARTICLE EXIT RESULTS)

            // .. ............................................................
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
                printf("-- Init              :  %3.2f ms \n", Enl00);
                printf("-- Save initial state:  %3.2f ms \n", EnlIn);
                printf("-- Propagation phase :  %3.2f ms \n", Enl0);
                printf("-- Find Max          :  %3.2f ms \n", Enl1);
                printf("-- Save final state  :  %3.2f ms \n", EnlFin);
                printf("-- Binning           :  %3.2f ms \n", Enl2);
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
            HANDLE_ERROR(cudaEventElapsedTime( &memset,
                start, MemorySet ));
            HANDLE_ERROR(cudaEventElapsedTime( &firstStep,
                start, Randomstep ));
            HANDLE_ERROR(cudaEventElapsedTime( &elapsedTime,
                start, stop ));
            printf("Time to Set Memory:  %3.1f ms \n", memset);
            printf("Time to create Rnd:  %3.1f ms (delta = %3.1f)\n", firstStep, firstStep - memset);
            printf("Time to execute   :  %3.1f ms (delta = %3.1f)\n", elapsedTime, elapsedTime - firstStep);
        }

        if constexpr (VERBOSE) {
            HANDLE_ERROR(cudaEventDestroy( start ));
            HANDLE_ERROR(cudaEventDestroy( MemorySet ));
            HANDLE_ERROR(cudaEventDestroy( Randomstep ));
            HANDLE_ERROR(cudaEventDestroy( stop ));
        }
    }
    // end of the multiple CPU thread pragma


    ////////////////////////////////////////////////////////////////
    //..... Exit results saving   ..................................
    ////////////////////////////////////////////////////////////////

    // Generate the YAML file name, following the old naming convention:
    // "<SimParameters.output_file_name>_matrix_<pid>.yaml"
    char yamlFilename[MaxCharinFileName];
    sprintf(yamlFilename, "%s_matrix_%lu.yaml", SimParameters.output_file_name, static_cast<unsigned long>(getpid()));

    try {
        write_results_yaml(yamlFilename, SimParameters);
        printf("Results saved to file: %s\n", yamlFilename);
    } catch (const std::exception &e) {
        std::cerr << "Error writing results to YAML file: " << e.what() << std::endl;
        return 1;
    }


    //  Save the summary histogram
    //  Free the dynamic memory

    // Save the rigidity histograms to txt file
    for (unsigned iR = 0; iR < SimParameters.NT; ++iR) {
        SaveTxt_histo(histo_filename, OldResults[iR].Nbins, OldResults[iR], VERBOSE_2);
    }

    /* save results to file .dat */
    FILE *pFile_Matrix = nullptr;
    char RAWMatrix_name[MaxCharinFileName];
    sprintf(RAWMatrix_name, "%s_matrix_%lu.dat", SimParameters.output_file_name,
            static_cast<unsigned long int>(getpid()));

    if constexpr (VERBOSE_2) fprintf(stdout, "Writing Output File: %s \n", RAWMatrix_name);
    pFile_Matrix = fopen(RAWMatrix_name, "w");

    if (pFile_Matrix == nullptr && VERBOSE_2) {
        fprintf(stderr, ERR_NoOutputFile);
        fprintf(stderr, "Writing to StandardOutput instead\n");
        pFile_Matrix = stdout;
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
                OldResults[itemp].Nregistered,
                OldResults[itemp].Nbins,
                OldResults[itemp].LogBin0_lowEdge,
                OldResults[itemp].DeltaLogR);
        if constexpr (VERBOSE) fprintf(pFile_Matrix, "# output distribution \n");

        for (int itNB = 0; itNB < OldResults[itemp].Nbins; itNB++) {
            fprintf(pFile_Matrix, "%e ", OldResults[itemp].BoundaryDistribution[itNB]);
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

    delete[] OldResults;

    if constexpr (VERBOSE) {
        // -- Save end time of simulation into log file
        time_t tim = time(nullptr);
        tm *local = localtime(&tim);
        printf("\nSimulation end at: %s  \n", asctime(local));
    }


    return cudaDeviceReset();
}
