#define MAINCU

// .. standard C
#include <cstdio>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions
#include <cstdlib>         // Supplies malloc(), calloc(), and realloc()
#include <unistd.h>         // Supplies EXIT_FAILURE, EXIT_SUCCESS
#include <sys/types.h>      // Typedef shortcuts like uint32_t and uint64_t
#include <sys/time.h>       // supplies time()

// .. multi-thread
#include <omp.h>

// math lib
#include <cmath>           // C math library
// .. CUDA specific
#include <curand_kernel.h>  // CUDA random number device library
#include <cuda_runtime.h>   // Device code management by providing implicit initialization, context management, and module management



// .. project specific
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
#ifndef MAX_DT
#define MAX_DT 50.                                        // max allowed value of time step
#endif
#ifndef MIN_DT
#define MIN_DT 0.01                                       // min allowed value of time step
#endif
#define TIMEOUT std::numeric_limits<float>::infinity()
// #define TIMEOUT 2000
#define NPOS 10
#define RBINS 100

// Debugging variables
#define VERBOSE 1
#define VERBOSE_2 1
#define VERBOSE_LOAD 0
#define SINGLE_CPU 0
#define HELMOD_LOAD 1
#define INITSAVE 0
#define FINALSAVE 0

// Datas variables
#define MaxCharinFileName   90

// -----------------------------------------------------------------
// ------------  Device Constant Variables declaration -------------
// -----------------------------------------------------------------
__constant__ SimulatedHeliosphere_t Heliosphere;
// Heliosphere properties include Local Interplanetary medium parameters
// __constant__ HeliosphereZoneProperties_t LIM[NMaxRegions]; // inner heliosphere
__constant__ HeliosheatProperties_t HS[NMaxRegions]; // heliosheat
// __constant__ float dev_Npart;
// __constant__ float min_dt;
// __constant__ float max_dt;
// __constant__ float timeout;

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

// Main Code
int main(int argc, char *argv[]) {
    // cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    ////////////////////////////////////////////////////////////////
    //..... Print Start time  ...................................
    // This part is for debug and performances tests
    ////////////////////////////////////////////////////////////////
    if constexpr (VERBOSE) {
        // -- Save initial time of simulation
        time_t tim = time(nullptr);
        tm *local = localtime(&tim);
        printf("\nSimulation started at: %s  \n", asctime(local));
    }
    ////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////
    //..... Initialize CPU threads   ...............................
    ////////////////////////////////////////////////////////////////

    //  Run as many CPU threads as there are CUDA devices
    //  each CPU thread controls a different device, processing its
    //  portion of the data.
    //  Initialize the global simulation parameter fron the input file
    //  Start execution time recording


    // Retrive GPUs infos and set the CPU multi threads
    int NGPUs;
    HANDLE_ERROR(cudaGetDeviceCount(&NGPUs)); // Count the available GPUs

    if (NGPUs < 1) {
        fprintf(stderr, "No CUDA capable devices were detected\n");
        exit(EXIT_FAILURE);
    }

    // Retrive the infos of alla the available GPUs and eventually print them
    cudaDeviceProp *GPUs_profile = DeviceInfo(NGPUs, VERBOSE_2);

    omp_set_num_threads(NGPUs); // create as many CPU threads as there are CUDA devices

    if constexpr (VERBOSE_2) {
        printf("\n");
        printf("----- Global CPU infos -----\n");
        printf("Number of host CPUs:\t%d\n", omp_get_num_procs());
    }

#if SINGLE_CPU
        omp_set_num_threads(1);   // setting 1 CPU thread for easier debugging
        NGPUs = 1;                // setting 1 GPU thread for easier debugging

        if (VERBOSE) {
            printf("WARNING: only 1 CPU managing only 1 GPU thread is instanziated, for easier debugging\n\n");
        }
#endif

    // Allocate the intial positions and rigidities into which load simulation configuration values
    InitialPositions_t InitialPositions;
    float *InitialRigidities;

    // Allocate simulation global parameters
    int NInitPos = 0;
    unsigned int NParts = 0;
    int NInitRig = 0;
    float RelativeBinAmplitude = 0;
    SimParameters_t SimParameters;

#if HELMOD_LOAD

    // NOTE: USING OLD STABLE 4_CoreCode_MultiGPU_MultiYear VERSION
    if (LoadConfigYaml(argc, argv, SimParameters, VERBOSE_LOAD) != EXIT_SUCCESS) {
        printf("Error while loading simulation parameters\n");
        exit(EXIT_FAILURE);
    }


    if (LoadConfigFile(argc, argv, SimParameters, VERBOSE_LOAD) != EXIT_SUCCESS) {
        printf("Error while loading simulation parameters\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the needed parameters for the new cosmica code
    NInitPos = static_cast<int>(SimParameters.NInitialPositions);
    NParts = SimParameters.NInitialPositions * SimParameters.Npart;
    InitialPositions = LoadInitPos(NParts, VERBOSE_LOAD);

    ////////////////////////////////////////////////////////////////
    //..... Rescale Heliosphere to an effective one  ...............
    ////////////////////////////////////////////////////////////////

    for (int ipos = 0; ipos < NInitPos; ipos++) {
        SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos] = SimParameters.HeliosphereToBeSimulated.
                RadBoundary_real[ipos];
        RescaleToEffectiveHeliosphere(SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos],
                                      SimParameters.InitialPosition[ipos]);

        if constexpr (VERBOSE_LOAD) {
            fprintf(stderr, "--- Zone %d \n", ipos);
            fprintf(
                stderr,
                "--- !! Effective Heliosphere --> effective boundaries: TS_nose=%f TS_tail=%f Rhp_nose=%f  Rhp_tail=%f \n",
                SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos].Rts_nose,
                SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos].Rts_tail,
                SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos].Rhp_nose,
                SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos].Rhp_tail);
            fprintf(stderr, "--- !! Effective Heliosphere --> new Source Position: r=%f th=%f phi=%f \n",
                    SimParameters.InitialPosition[ipos].r, SimParameters.InitialPosition[ipos].th,
                    SimParameters.InitialPosition[ipos].phi);
        }

        // Copy initial positions from SimParameters to the CPU InitialPositions_t
        InitialPositions.r[ipos] = SimParameters.InitialPosition[ipos].r;
        InitialPositions.th[ipos] = SimParameters.InitialPosition[ipos].th;
        InitialPositions.phi[ipos] = SimParameters.InitialPosition[ipos].phi;
    }

    NInitRig = static_cast<int>(SimParameters.NT);
    InitialRigidities = LoadInitRigidities(NInitRig, VERBOSE_LOAD);

    for (int iR = 0; iR < NInitRig; iR++) {
        InitialRigidities[iR] = SimParameters.Tcentr[iR];
    }

    // relative (respect 1.) amplitude of Energy bin used as X axis in BoundaryDistribution  --> delta T = T*RelativeBinAmplitude
    RelativeBinAmplitude = SimParameters.RelativeBinAmplitude;

    // Load the global simulation parameters with new cosmica-GC method
#else
        // Load the initial positions and particle number to simulate
        InitialPositions = LoadInitPos(NPOS, VERBOSE_LOAD);
        NInitPos = NPOS;
        NParts = NPARTS;

        // Load rigidities to be simulated
        InitialRigidities = LoadInitRigidities(RBINS, VERBOSE_LOAD);
        NInitRig = RBINS;

        // relative (respect 1.) amplitude of Energy bin used as X axis in BoundaryDistribution  --> delta T = T*RelativeBinAmplitude
        RelativeBinAmplitude = 0.00855;

    // Load the global simulation parameters with old HelMod method
    /*
    typedef struct SimParameters_t {                                                    // Place here all simulation variables
        char  output_file_name[struct_string_lengh]="SimTest";
        unsigned long      Npart=5000;                                  // number of event to be simulated
        unsigned char      NT;                                          // number of bins of energies to be simulated
        unsigned char      NInitialPositions=0;                         // number of initial positions -> this number represent also the number of Carrington rotation that
        float              *Tcentr;                                     // array of energies to be simulated
        vect3D_t           *InitialPosition;                            // initial position
        PartDescription_t  IonToBeSimulated;                            // Ion to be simulated
        MonteCarloResult_t *Results;                                    // output of the code
        float RelativeBinAmplitude = 0.00855 ;                          // relative (respect 1.) amplitude of Energy bin used as X axis in BoundaryDistribution  --> delta T = T*RelativeBinAmplitude
        SimulatedHeliosphere_t HeliosphereToBeSimulated;                // Heliosphere properties for the simulation
        HeliosphereZoneProperties_t prop_medium[NMaxRegions];           // PROPerties of the interplanetary MEDIUM - Heliospheric Parameters in each Heliospheric Zone
        HeliosheatProperties_t prop_Heliosheat[NMaxRegions];            // Properties of Heliosheat
    } SimParameters_t;
    */
#endif

    // Allocation of the output results for all the rigidities
    auto *Results = static_cast<struct MonteCarloResult_t *>(malloc(NInitRig * sizeof(MonteCarloResult_t)));

    // .. Results saving files

    // Initial and final results files
    char file_trivial[8];
    sprintf(file_trivial, "");

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
        LaunchParam_t prop_launch_param = RoundNpart(NParts, device_prop, VERBOSE_2, SetWarpPerBlock, 1);

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
        auto RandStates = AllocateManagedSafe<curandStatePhilox4_32_10_t>(NParts);
        unsigned long Rnd_seed = SimParameters.RandomSeed == 0
                                     ? getpid() + time(nullptr) + gpu_id
                                     : SimParameters.RandomSeed;
        cudaDeviceSynchronize();
        init_rdmgenerator<<<prop_launch_param.blocks, prop_launch_param.threads>>>(RandStates.get(), Rnd_seed);
        cudaDeviceSynchronize();

        if constexpr (VERBOSE) {
            //.. capture the time from GPU
            HANDLE_ERROR(cudaEventRecord( Randomstep, nullptr ));
            HANDLE_ERROR(cudaEventSynchronize( Randomstep ));
            if constexpr (VERBOSE_2) {
                fprintf(stdout, "--- Random Generator Seed: %lu \n", Rnd_seed);
            }
        }

        // .. copy heliosphere parameters to Device Constant Memory
        CopyToConstant(Heliosphere, &SimParameters.HeliosphereToBeSimulated);
        // CopyToConstant(LIM, &SimParameters.prop_medium);
        CopyToConstant(HS, &SimParameters.prop_Heliosheat);


        // HeliosphereZoneProperties_t LIM[NMaxRegions];
        auto LIM = AllocateManagedSafe<HeliosphereZoneProperties_t>(NMaxRegions);
        cudaMemcpy(LIM.get(), &SimParameters.prop_medium, sizeof(SimParameters.prop_medium), cudaMemcpyDefault);


        // Allocate the initial variables and allocate on device
        ThreadQuasiParticles_t QuasiParts = AllocateQuasiParticles(NParts);

        // Period along which CR are integrated and the corresponding period indecies
        ThreadIndexes indexes = AllocateIndex(NParts);
        unsigned int ns = 1, ni = SimParameters.NInitialPositions, np = 1, nx = SimParameters.Npart;
        //TODO: check ordering for adjacency in warp
        for (unsigned int s = 0; s < ns; ++s) {
            for (unsigned int i = 0; i < ni; ++i) {
                for (unsigned int p = 0; p < np; ++p) {
                    for (unsigned int x = 0; x < nx; ++x) {
                        unsigned int idx = x + nx * (p + np * (i + ni * s));
                        indexes.simulation[idx] = s;
                        indexes.period[idx] = i;
                        indexes.particle[idx] = p;
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
        for (unsigned int iR = gpu_id; iR < NInitRig; iR += NGPUs) {
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

            // GPU propagation kernel execution parameters debugging
            if constexpr (VERBOSE_2) {
                printf("\n-- Cycle on rigidity[%d]: %.2f \n", iR, InitialRigidities[iR]);
                printf("Quasi-particles propagation kernel launched\n");
                printf("Number of quasi-particles: %d\n", NParts);
                printf("Number of blocks: %d\n", prop_launch_param.blocks);
                printf("Number of threads per block: %d\n", prop_launch_param.threads);
                printf("Number of shared memory bytes per block: %d\n", prop_launch_param.smem);
            }

            // Initialize the particle starting rigidities
            for (int iPart = 0; iPart < NParts; iPart++) {
                QuasiParts.r[iPart] = InitialPositions.r[indexes.period[iPart]];
                QuasiParts.th[iPart] = InitialPositions.th[indexes.period[iPart]];
                QuasiParts.phi[iPart] = InitialPositions.phi[indexes.period[iPart]];
                QuasiParts.R[iPart] = InitialRigidities[iR];
                QuasiParts.t_fly[iPart] = 0;
            }


            // Allocate the array for the partial rigidities maxima and final maximum
            auto Maxs = AllocateManagedSafe<float>(prop_launch_param.blocks);


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
            HeliosphericProp<<<prop_launch_param.blocks, prop_launch_param.threads, prop_launch_param.smem>>>
            (NParts, MIN_DT, MAX_DT, TIMEOUT, QuasiParts, indexes, LIM.get(), RandStates.get(),
             Maxs.get());
            cudaDeviceSynchronize();

            if constexpr (VERBOSE) {
                HANDLE_ERROR(cudaEventRecord( Cycle_step0, nullptr ));
                HANDLE_ERROR(cudaEventSynchronize( Cycle_step0 ));
            }

            if constexpr (VERBOSE_2) {
                fprintf(stdout, "--- Max values: ");
                for (int itemp = 0; itemp < prop_launch_param.blocks; itemp++) {
                    fprintf(stdout, "%.2f ", Maxs[itemp]);
                }
            }

            // ->then finalize on CPU
            for (int itemp = 1; itemp < prop_launch_param.blocks; itemp++) {
                if (Maxs[0] < Maxs[itemp]) {
                    Maxs[0] = Maxs[itemp];
                }
            }
            if constexpr (VERBOSE_2)
                fprintf(stdout, "\n--- RMin = %.3f Rmax = %.3f \n", InitialRigidities[iR],
                        Maxs[0]);

            if (Maxs[0] < SimParameters.Tcentr[iR]) {
                printf("PROBLEMA: the max exiting rigidity is smaller than initial one\n");
                continue;
            }

            if constexpr (VERBOSE) {
                HANDLE_ERROR(cudaEventRecord( Cycle_step1, nullptr ));
                HANDLE_ERROR(cudaEventSynchronize( Cycle_step1 ));
            }

            if constexpr (FINALSAVE) {
                // host final states for specific energy
                ThreadQuasiParticles_t host_final_QuasiParts = AllocateQuasiParticles(NParts);

                SaveTxt_part(final_filename, NParts, host_final_QuasiParts, Maxs[0], VERBOSE_2);

                // Free the host particle variable for the energy on which the cycle is running
                free(host_final_QuasiParts.r);
                free(host_final_QuasiParts.th);
                free(host_final_QuasiParts.phi);
                free(host_final_QuasiParts.R);
                free(host_final_QuasiParts.t_fly);
                // free(host_final_QuasiParts.alphapath);
            }

            if constexpr (VERBOSE) {
                HANDLE_ERROR(cudaEventRecord( FinalSave, nullptr ));
                HANDLE_ERROR(cudaEventSynchronize( FinalSave ));
            }

            // Definition of histogram binning as a fraction of the bin border (DeltaT=T*RelativeBinAmplitude)
            float DeltaLogR = log10f(1.f + RelativeBinAmplitude);
            float LogBin0_lowEdge = log10f(InitialRigidities[iR]) - DeltaLogR / 2.f;
            float Bin0_lowEdge = powf(10, LogBin0_lowEdge); // first LowEdge Bin

            Results[iR].Nbins = ceil(log10(Maxs[0] / Bin0_lowEdge) / DeltaLogR);
            Results[iR].LogBin0_lowEdge = LogBin0_lowEdge;
            Results[iR].DeltaLogR = DeltaLogR;

            // .. save to histogram ..........................................
            // Partial block histogram allocation
            auto PartialHistos = AllocateManagedSafe<float>(Results[iR].Nbins * prop_launch_param.blocks);

            // Final merged histogram allocation
            Results[iR].BoundaryDistribution = AllocateManaged<float>(Results[iR].Nbins);

            auto Nfailed = AllocateManagedSafe<int>(1, 0);

            // Partial histogram atomoic sum on GPU
            cudaDeviceSynchronize();
            histogram_atomic<<<prop_launch_param.blocks, prop_launch_param.threads>>>(
                QuasiParts.R, LogBin0_lowEdge, DeltaLogR, Results[iR].Nbins, NParts,
                PartialHistos.get(), Nfailed.get());
            cudaDeviceSynchronize();

            // Failed quasi-particle propagation count
            Results[iR].Nregistered = NParts - Nfailed[0];

            if constexpr (VERBOSE_2) {
                fprintf(stdout, "-- Eventi computati : %d \n", NParts);
                fprintf(stdout, "-- Eventi falliti   : %d \n", Nfailed[0]);
                fprintf(stdout, "-- Eventi registrati: %lu \n", Results[iR].Nregistered);
            }

            int histo_Nblocchi = ceil_int(Results[iR].Nbins, prop_launch_param.threads);

            // Merge of the partial histograms and copy to the host
            cudaDeviceSynchronize();
            histogram_accum<<<histo_Nblocchi, prop_launch_param.threads>>>(
                PartialHistos.get(), Results[iR].Nbins, prop_launch_param.blocks, Results[iR].BoundaryDistribution);
            cudaDeviceSynchronize();


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

    //  Save the summary histogram
    //  Free the dynamic memory

    // Save the rigidity histograms to txt file
    for (int iR = 0; iR < NInitRig; iR++) {
        SaveTxt_histo(histo_filename, Results[iR].Nbins, Results[iR], VERBOSE_2);
    }

    /* save results to file .dat */
#if HELMOD_LOAD
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

    for (int itemp = 0; itemp < SimParameters.NT; itemp++) {
        if constexpr (VERBOSE) {
            fprintf(pFile_Matrix, "######  Bin %d \n", itemp);
            fprintf(pFile_Matrix,
                    "# Egen, Npart Gen., Npart Registered, Nbin output, log10(lower edge bin 0), Bin amplitude (in log scale)\n");
        }

        fprintf(pFile_Matrix, "%f %u %lu %d %f %f \n", SimParameters.Tcentr[itemp],
                SimParameters.Npart,
                Results[itemp].Nregistered,
                Results[itemp].Nbins,
                Results[itemp].LogBin0_lowEdge,
                Results[itemp].DeltaLogR);
        if constexpr (VERBOSE) fprintf(pFile_Matrix, "# output distribution \n");

        for (int itNB = 0; itNB < Results[itemp].Nbins; itNB++) {
            fprintf(pFile_Matrix, "%e ", Results[itemp].BoundaryDistribution[itNB]);
        }


        fprintf(pFile_Matrix, "\n");
        fprintf(pFile_Matrix, "#\n"); // <--- dummy line to separate results
    }

    fflush(pFile_Matrix);
    fclose(pFile_Matrix);
#endif

    // Free of the initial simulation variables
    free(InitialPositions.r);
    free(InitialPositions.th);
    free(InitialPositions.phi);
    free(InitialRigidities);

    free(SimParameters.Tcentr);

    free(GPUs_profile);

    free(Results);

    if constexpr (VERBOSE) {
        // -- Save end time of simulation into log file
        time_t tim = time(nullptr);
        tm *local = localtime(&tim);
        printf("\nSimulation end at: %s  \n", asctime(local));
    }


    return cudaDeviceReset();
}
