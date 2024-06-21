#define MAINCU

// .. standard C
#include <stdio.h>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions
#include <stdlib.h>         // Supplies malloc(), calloc(), and realloc()
#include <unistd.h>         // Supplies EXIT_FAILURE, EXIT_SUCCESS
#include <libgen.h>         // Supplies the basename() function 
#include <errno.h>          // Defines the external errno variable and all the values it can take on
#include <string.h>         // Supplies memcpy(), memset(), and the strlen() family of functions
#include <getopt.h>         // Supplies external optarg, opterr, optind, and getopt() function
#include <sys/types.h>      // Typedef shortcuts like uint32_t and uint64_t
#include <sys/time.h>       // supplies time()

// .. multi-thread
#include <omp.h>

// math lib
#include <math.h>           // C math library
#include <limits.h>         // numerical C limits 
// .. CUDA specific
#include <curand.h>         // CUDA random number host library
#include <curand_kernel.h>  // CUDA random number device library
#include <cuda_runtime.h>   // Device code management by providing implicit initialization, context management, and module management

// .. project specific
#include "VariableStructure.cuh"
#include "SDECoeffs.cuh"
#include "LoadConfiguration.cuh"
#include "HeliosphericPropagation.cuh"
#include "HeliosphereLocation.cuh"
#include "GenComputation.cuh"
#include "HistoComputation.cuh"
#include "GPUManage.cuh"
#include "Histogram.cuh"

// .. old HelMod code
#include "HelModVariableStructure.cuh"
#include "HelModLoadConfiguration.cuh"
#include "MagneticDrift.cuh"
#include "SolarWind.cuh"
#include "DiffusionModel.cuh"
#include "HeliosphereModel.cuh"

// Track the errors
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define ERR_NoOutputFile "ERROR: output file cannot be open, do you have writing permission?\n"

// Simulation iperparameters definition
#define WARPSIZE 32
#ifndef SetWarpPerBlock
  #define SetWarpPerBlock -1                                          // number of warp so be submitted -- modify this value to find the best performance
#endif
#define NPARTS 5000
#ifndef MAX_DT
  #define MAX_DT 50.                                        // max allowed value of time step
#endif
#ifndef MIN_DT
  #define MIN_DT 0.01                                       // min allowed value of time step
#endif
#define TIMEOUT std::numeric_limits<float>::infinity()
#define NPOS 10
#define RBINS 100

// Debugging variables
#define VERBOSE 1
#define VERBOSE_2 1
#define VERBOSE_LOAD 3
#define SINGLE_CPU 1
#define HELMOD_LOAD 1
#define INITSAVE 1
#define FINALSAVE 1
#define TRIVIAL 0
#define NVIDIA_HIST 0

// Datas variables
#define MaxCharinFileName   90

// -----------------------------------------------------------------
// ------------  Device Constant Variables declaration -------------
// -----------------------------------------------------------------
__constant__ SimulatedHeliosphere_t      Heliosphere;            // Heliosphere properties include Local Interplanetary medium parameters
__constant__ HeliosphereZoneProperties_t LIM[NMaxRegions];       // inner heliosphere
__constant__ HeliosheatProperties_t      HS[NMaxRegions];        // heliosheat
// __constant__ float dev_Npart;
// __constant__ float min_dt;
// __constant__ float max_dt;
// __constant__ float timeout;


// Main Code
int main(int argc, char* argv[]) {

    ////////////////////////////////////////////////////////////////
    //..... Print Start time  ...................................
    // This part is for debug and performances tests
    ////////////////////////////////////////////////////////////////
    if (VERBOSE)
    {
        // -- Save initial time of simulation
        time_t tim =time(NULL);
        struct tm *local = localtime(&tim);
        printf("\nSimulation started at: %s  \n",asctime(local));
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
    HANDLE_ERROR(cudaGetDeviceCount(&NGPUs));   // Count the available GPUs
    
    if (NGPUs<1) {
        fprintf(stderr,"No CUDA capable devices were detected\n");
        exit(EXIT_FAILURE);
    }

    // Retrive the infos of alla the available GPUs and eventually print them
    cudaDeviceProp* GPUs_profile = DeviceInfo(NGPUs, VERBOSE);

    omp_set_num_threads(NGPUs);                 // create as many CPU threads as there are CUDA devices
    unsigned int num_cpu_threads = omp_get_num_threads();  // numero totale di CPU-thread allocated

    printf("\n");
    printf("----- Global CPU infos -----\n" );
    printf("Number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("Number of host CPUs allocated:\t%d\n", num_cpu_threads);

    #if SINGLE_CPU
        omp_set_num_threads(1);   // setting 1 CPU thread for easier debugging
        NGPUs = 1;                // setting 1 GPU thread for easier debugging

        if (VERBOSE) {
            printf("WARNING: only 1 CPU managing only 1 GPU thread is instanziated, for easier debugging\n\n");
        }
    #endif

    // Allocate the intial positions and rigidities into which load simulation configuration values
    struct InitialPositions_t InitialPositions;
    float* InitialRigidities;

    // Allocate simulation global parameters
    int NInitPos = 0;
    int NParts = 0;
    int NInitRig = 0;
    float RelativeBinAmplitude = 0;
    struct SimParameters_t SimParameters;
    struct PartDescription_t pt;

    #if HELMOD_LOAD

        // NOTE: USING OLD STABLE 4_CoreCode_MultiGPU_MultiYear VERSION
        if (Load_Configuration_File(argc, argv, SimParameters, VERBOSE) != EXIT_SUCCESS) {
            printf("Error while loading simulation parameters\n");
            exit(EXIT_FAILURE);
        }

        // Initialize the needed parameters for the new cosmica code
        NInitPos = (int)SimParameters.NInitialPositions;
        NParts = (int)SimParameters.Npart;
        InitialPositions = LoadInitPos(NParts, VERBOSE);
        pt = SimParameters.IonToBeSimulated;

        ////////////////////////////////////////////////////////////////
        //..... Rescale Heliosphere to an effective one  ...............
        ////////////////////////////////////////////////////////////////

        for (int ipos=0; ipos<NInitPos; ipos++) {
            SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos] = SimParameters.HeliosphereToBeSimulated.RadBoundary_real[ipos];
            RescaleToEffectiveHeliosphere(SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos], SimParameters.InitialPosition[ipos]);
            
            if (VERBOSE_2){
                fprintf(stderr,"--- Zone %d \n", ipos);
                fprintf(stderr,"--- !! Effective Heliosphere --> effective boundaries: TS_nose=%f TS_tail=%f Rhp_nose=%f  Rhp_tail=%f \n", SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos].Rts_nose,SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos].Rts_tail,SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos].Rhp_nose,SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos].Rhp_tail);
                fprintf(stderr,"--- !! Effective Heliosphere --> new Source Position: r=%f th=%f phi=%f \n", SimParameters.InitialPosition[ipos].r, SimParameters.InitialPosition[ipos].th,SimParameters.InitialPosition[ipos].phi);
            }
        
            // Copy initial positions from SimParameters to the CPU InitialPositions_t
            InitialPositions.r[ipos] = SimParameters.InitialPosition[ipos].r;
            InitialPositions.th[ipos] = SimParameters.InitialPosition[ipos].th;
            InitialPositions.phi[ipos] = SimParameters.InitialPosition[ipos].phi;
        }

        NInitRig = (int)SimParameters.NT;
        InitialRigidities = LoadInitRigidities(NInitRig, VERBOSE);

        for (int iR=0; iR<NInitRig; iR++) {
            InitialRigidities[iR] = SimParameters.Tcentr[iR];
        }

        // relative (respect 1.) amplitude of Energy bin used as X axis in BoundaryDistribution  --> delta T = T*RelativeBinAmplitude
        RelativeBinAmplitude = SimParameters.RelativeBinAmplitude;
    
    // Load the global simulation parameters with new cosmica-GC method
    #else
        // Load the initial positions and particle number to simulate
        InitialPositions = LoadInitPos(NPOS, VERBOSE_2);
        NInitPos = NPOS;
        NParts = NPARTS;

        // Load rigidities to be simulated
        InitialRigidities = LoadInitRigidities(RBINS, VERBOSE_2);
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
    struct MonteCarloResult_t* Results = (struct MonteCarloResult_t*)malloc(NInitRig*sizeof(MonteCarloResult_t));

    // .. Results saving files

    // Initial and final results files
    char file_trivial[8];
    #if TRIVIAL
        sprintf(file_trivial, "trivial_");
    #else
        sprintf(file_trivial, "");
    #endif

    char init_filename[20];
    sprintf(init_filename, "%sprop_in.txt", file_trivial);
    char final_filename[ReadingStringLenght];
    sprintf(final_filename, "%s_%sprop_out.txt", SimParameters.output_file_name, file_trivial);
    // Clean previous files
    if (remove(init_filename) != 0 || remove(final_filename) != 0) printf("Error deleting the old propagation files or it does not exist\n");
    else printf("Old propagation files deleted successfully\n");

    // Initial and final results files
    char histo_filename[20];
    sprintf(histo_filename, "%sR_histo.txt", file_trivial);
    // Clean previous files
    if (remove(histo_filename) != 0) printf("Error deleting the old histogram files or it does not exist\n");
    else printf("Old histogram files deleted successfully\n");
    printf( "-- \n\n" );


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
        unsigned int cpu_thread_id = omp_get_thread_num();     // identificativo del CPU-thread
        int gpu_id = cpu_thread_id % NGPUs;              // seleziona la id della GPU da usare. "% num_gpus" allows more CPU threads than GPU devices
        HANDLE_ERROR(cudaSetDevice(gpu_id));                   // seleziona la GPU

        if (VERBOSE) {
            printf( "----- Individual CPU infos -----\n" );
            printf("-- CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);
            printf( "-- \n\n" );
        }


        // Retrive information from the set GPU
        cudaDeviceProp device_prop = GPUs_profile[gpu_id];

        // Rounding the number of particle and calculating threads, blocks and share memory to acheive the maximum usage of the GPUs
        LaunchParam_t prop_launch_param = RoundNpart(NParts, device_prop, VERBOSE, SetWarpPerBlock);

        ////////////////////////////////////////////////////////////////
        //..... capture the start time of GPU part .....................
        //      This part is for debug and performances tests
        ////////////////////////////////////////////////////////////////
        cudaEvent_t     start,MemorySet,Randomstep, stop;
        cudaEvent_t     Cycle_start,Cycle_step00,Cycle_step0,Cycle_step1,Cycle_step2, InitialSave, FinalSave;
        if (VERBOSE){
        HANDLE_ERROR( cudaEventCreate( &start ) );
        HANDLE_ERROR( cudaEventCreate( &MemorySet ) );
        HANDLE_ERROR( cudaEventCreate( &Randomstep ) );
        HANDLE_ERROR( cudaEventCreate( &stop ) );
        HANDLE_ERROR( cudaEventRecord( start, 0 ) );
        }
        ////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////
        //..... GPU execution initialization   .........................
        ////////////////////////////////////////////////////////////////

        //  Set pseudo random number generator seeds
        //  Device memory allocation and threads starting positions


        // .. Initialize random generator
        curandStatePhilox4_32_10_t *dev_RndStates;
        HANDLE_ERROR(cudaMalloc((void **)&dev_RndStates, prop_launch_param.Npart*sizeof(curandStatePhilox4_32_10_t)));
        unsigned long Rnd_seed=getpid()+time(NULL)+gpu_id;
        init_rdmgenerator<<<prop_launch_param.blocks, prop_launch_param.threads>>>(dev_RndStates, Rnd_seed);
        cudaDeviceSynchronize();

        if (VERBOSE){
            //.. capture the time from GPU
            HANDLE_ERROR( cudaEventRecord( Randomstep, 0 ) );
            HANDLE_ERROR( cudaEventSynchronize( Randomstep ) );
            if (VERBOSE>=VERBOSE_med){
              fprintf(stdout,"--- Random Generator Seed: %lu \n",Rnd_seed);
            }
        }

        // .. copy heliosphere parameters to Device Constant Memory
        cudaMemcpyToSymbol(Heliosphere, &SimParameters.HeliosphereToBeSimulated, sizeof(SimulatedHeliosphere_t));
        cudaMemcpyToSymbol(LIM, &SimParameters.prop_medium   , NMaxRegions*sizeof(HeliosphereZoneProperties_t));
        cudaMemcpyToSymbol(HS, &SimParameters.prop_Heliosheat, NMaxRegions*sizeof(HeliosheatProperties_t));
        // cudaMemcpyToSymbol(dev_Npart, &prop_launch_param.Npart, sizeof(float));
        // cudaMemcpyToSymbol(min_dt, &MIN_DT, sizeof(float));
        // cudaMemcpyToSymbol(max_dt, &MAX_DT, sizeof(float));
        // cudaMemcpyToSymbol(timeout, &TIMEOUT, sizeof(float));
      
        // allocate on host
        struct QuasiParticle_t host_QuasiParts = InitQuasiPart_mem(prop_launch_param.Npart, 0, VERBOSE_2);   // host initial state of propagation kernel

        // Allocate the initial variables and allocate on device
        struct QuasiParticle_t dev_QuasiParts = InitQuasiPart_mem(prop_launch_param.Npart, 1, VERBOSE_2);    // device input/output of propagation kernel

        // Period along which CR are integrated and the corresponding period indecies
        int* dev_PeriodIndexes;
        HANDLE_ERROR(cudaMalloc((void**)&dev_PeriodIndexes, prop_launch_param.Npart*sizeof(int)));
        cudaDeviceSynchronize();

        int* host_PeriodIndexes = (int*)malloc(prop_launch_param.Npart*sizeof(int));

        // initialize the host array
        // The particle simulated in the kernel are distributed between the initial positions using the period index
        for(int iPart=0; iPart<prop_launch_param.Npart; iPart++) {
            int PeriodIndex = floor_int(iPart*NInitPos, prop_launch_param.Npart);
            host_PeriodIndexes[iPart]    = PeriodIndex;
            host_QuasiParts.r[iPart]     = InitialPositions.r[PeriodIndex];
            host_QuasiParts.th[iPart]    = InitialPositions.th[PeriodIndex];
            host_QuasiParts.phi[iPart]   = InitialPositions.phi[PeriodIndex];
            // host_QuasiParts.R[iPart]     = InitialRigidities[floor_int(iPart*NInitRig, prop_launch_param.Npart)];
            host_QuasiParts.t_fly[iPart] = 0;
            // host_QuasiParts.alphapath[iPart] = 0;
        }

        // copy host_PeriodIndexes to dev_PeriodIndexes and free memory
        HANDLE_ERROR(cudaMemcpy(dev_PeriodIndexes, host_PeriodIndexes, prop_launch_param.Npart*sizeof(int), cudaMemcpyHostToDevice));

        // Recording the setting memory execution time
        if (VERBOSE){
            HANDLE_ERROR( cudaEventRecord( MemorySet, 0 ) );
            HANDLE_ERROR( cudaEventSynchronize( MemorySet ) );
          }      

        ////////////////////////////////////////////////////////////////
        //..... GPU perticle propagation   .............................
        ////////////////////////////////////////////////////////////////

        //  Initialization of the cycle on rigidities bins (for all the positions)
        //  Launch of the GPU propagation kernel (computing diffusion 
        //  coefficients and solving stochastic differential equations)
        //  Build the exit energy histogram


        // Cycle on rigidity bins distributing their execution between the active CPU threads
        for (int iR=gpu_id; iR<NInitRig ; iR+=NGPUs) {

            if (VERBOSE){
                HANDLE_ERROR( cudaEventCreate( &Cycle_start ) );
                HANDLE_ERROR( cudaEventCreate( &Cycle_step00 ) );
                HANDLE_ERROR( cudaEventCreate( &Cycle_step0 ) );
                HANDLE_ERROR( cudaEventCreate( &Cycle_step1 ) );
                HANDLE_ERROR( cudaEventCreate( &Cycle_step2 ) );
                HANDLE_ERROR( cudaEventCreate( &InitialSave ) );
                HANDLE_ERROR( cudaEventCreate( &FinalSave ) );
                HANDLE_ERROR( cudaEventRecord( Cycle_start, 0 ) );
              }
            
            // GPU propagation kernel execution parameters debugging
            if (VERBOSE) {
                printf("\n-- Cycle on rigidity[%d]: %.2f \n", iR , InitialRigidities[iR]);
                printf("Quasi-particles propagation kernel launched\n");
                printf("Number of quasi-particles: %d\n", prop_launch_param.Npart);
                printf("Number of blocks: %d\n", prop_launch_param.blocks);
                printf("Number of threads per block: %d\n", prop_launch_param.threads);
                printf("Number of shared memory bytes per block: %d\n", prop_launch_param.smem);
            }
            
            // Initialize the particle starting rigidities
            for(int iPart=0; iPart<prop_launch_param.Npart; iPart++) {
                host_QuasiParts.R[iPart] = InitialRigidities[iR];
            }

            // copy host initial propagation states to device quasi particle states
            HANDLE_ERROR(cudaMemcpy(dev_QuasiParts.r, host_QuasiParts.r, prop_launch_param.Npart*sizeof(float), cudaMemcpyHostToDevice));
            HANDLE_ERROR(cudaMemcpy(dev_QuasiParts.th, host_QuasiParts.th, prop_launch_param.Npart*sizeof(float), cudaMemcpyHostToDevice));
            HANDLE_ERROR(cudaMemcpy(dev_QuasiParts.phi, host_QuasiParts.phi, prop_launch_param.Npart*sizeof(float), cudaMemcpyHostToDevice));
            HANDLE_ERROR(cudaMemcpy(dev_QuasiParts.R, host_QuasiParts.R, prop_launch_param.Npart*sizeof(float), cudaMemcpyHostToDevice));
            HANDLE_ERROR(cudaMemcpy(dev_QuasiParts.t_fly, host_QuasiParts.t_fly, prop_launch_param.Npart*sizeof(float), cudaMemcpyHostToDevice));
            // HANDLE_ERROR(cudaMemcpy(dev_QuasiParts.alphapath, host_QuasiParts.alphapath, prop_launch_param.Npart*sizeof(float), cudaMemcpyHostToDevice));

            // Allocate the array for the partial rigidities maxima and final maximum
            float* dev_maxs;
            HANDLE_ERROR(cudaMalloc((void **) &dev_maxs, prop_launch_param.blocks*sizeof(float)));

            #if NVIDIA_HIST
                float* dev_Rmax;
                HANDLE_ERROR(cudaMalloc((void **) &dev_Rmax, 2*sizeof(float)));
                host_Rmax = (float*)malloc(2*sizeof(float));
                cudaDeviceSynchronize();

            #else
                float* host_Rmax;
                host_Rmax = (float*)malloc(prop_launch_param.blocks*sizeof(float));
                cudaDeviceSynchronize();
            #endif


            if (VERBOSE){
                HANDLE_ERROR( cudaEventRecord( Cycle_step00, 0 ) );
                HANDLE_ERROR( cudaEventSynchronize( Cycle_step00 ) );
              }        

            // Saving the initial particles parameters into a txt file for debugging
            if (INITSAVE) {
                SaveTxt_part(init_filename, prop_launch_param.Npart, host_QuasiParts, host_Rmax[0], VERBOSE_2);
            }

            if (VERBOSE){
                HANDLE_ERROR( cudaEventRecord( InitialSave, 0 ) );
                HANDLE_ERROR( cudaEventSynchronize( InitialSave ) );
            }


            // Heliosphere propagation kernel
            // and local max rigidity search inside the block
            HeliosphericProp<<<prop_launch_param.blocks, prop_launch_param.threads, prop_launch_param.smem>>>
            (prop_launch_param.Npart, MIN_DT, MAX_DT, TIMEOUT, dev_QuasiParts, dev_PeriodIndexes, pt, dev_RndStates, dev_maxs);

            // (taking into account the different possible block dimension template of BlockMax execution)
            /* switch (prop_launch_param.threads) {
                case 512:
                HeliosphericProp<512><<<prop_launch_param.blocks, prop_launch_param.threads, prop_launch_param.smem>>>
                (prop_launch_param.Npart, MAX_DT, TIMEOUT, dev_QuasiParts, dev_RndStates, dev_maxs);
                break;
                case 256:
                HeliosphericProp<256><<<prop_launch_param.blocks, prop_launch_param.threads, prop_launch_param.smem>>>
                (prop_launch_param.Npart, MAX_DT, TIMEOUT, dev_QuasiParts, dev_RndStates, dev_maxs);
                break;
                case 128:
                HeliosphericProp<128><<<prop_launch_param.blocks, prop_launch_param.threads, prop_launch_param.smem>>>
                (prop_launch_param.Npart, MAX_DT, TIMEOUT, dev_QuasiParts, dev_RndStates, dev_maxs);
                break;
                case 64:
                HeliosphericProp< 64><<<prop_launch_param.blocks, prop_launch_param.threads, prop_launch_param.smem>>>
                (prop_launch_param.Npart, MAX_DT, TIMEOUT, dev_QuasiParts, dev_RndStates, dev_maxs);
                break;
                case 32:
                HeliosphericProp< 32><<<prop_launch_param.blocks, prop_launch_param.threads, prop_launch_param.smem>>>
                (prop_launch_param.Npart, MAX_DT, TIMEOUT, dev_QuasiParts, dev_RndStates, dev_maxs);
                break;
            } */
            
            cudaDeviceSynchronize();

            if (VERBOSE) {
                HANDLE_ERROR( cudaEventRecord( Cycle_step0, 0 ) );
                HANDLE_ERROR( cudaEventSynchronize( Cycle_step0 ) );
            }        
        
            #if NVIDIA_HIST
                // Global max rigidity search from partial maxima
                unsigned int GridMax_threads = ceil_int(prop_launch_param.blocks/2, (device_prop.warpSize))*(device_prop.warpSize);
                GridMax<<<2, 2*GridMax_threads, GridMax_threads*sizeof(float)>>>(prop_launch_param.blocks, dev_maxs, dev_Rmax);
                
                // (taking into account the different possible block dimension template of GridMax execution)
                /* switch (prop_launch_param.blocks) {
                    case 512:
                        GridMax<512><<<1, prop_launch_param.blocks, prop_launch_param.blocks*sizeof(float)>>>(dev_maxs, dev_Rmax);
                        break;
                    case 256:
                        GridMax<256><<<1, prop_launch_param.blocks, prop_launch_param.blocks*sizeof(float)>>>(dev_maxs, dev_Rmax);
                        break;
                    case 128:
                        GridMax<128><<<1, prop_launch_param.blocks, prop_launch_param.blocks*sizeof(float)>>>(dev_maxs, dev_Rmax);
                        break;
                    case 64:
                        GridMax< 64><<<1, prop_launch_param.blocks, prop_launch_param.blocks*sizeof(float)>>>(dev_maxs, dev_Rmax);
                        break;
                    case 32:
                        GridMax< 32><<<1, prop_launch_param.blocks, prop_launch_param.blocks*sizeof(float)>>>(dev_maxs, dev_Rmax);
                        break;
                } */

                // Copy the final maximum rigidity to host and free partial maxima array memory
                HANDLE_ERROR(cudaMemcpy(host_Rmax, dev_Rmax, 2*sizeof(float), cudaMemcpyDeviceToHost));
                cudaFree(dev_Rmax);

                // Finalization of the maximum rigidity search on CPU
                if (host_Rmax[0]<host_Rmax[1]) host_Rmax[0] = host_Rmax[1];

            #else

                cudaMemcpy(host_Rmax, dev_maxs, prop_launch_param.blocks*sizeof(float), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();    

                if (VERBOSE_2){
                    fprintf(stdout,"--- Max values: ");
                    for (int itemp=0; itemp<prop_launch_param.blocks; itemp++) {
                        fprintf(stdout,"%.2f ", host_Rmax[itemp]);
                    }
                    fprintf(stdout,"\n");
                    fprintf(stdout,"--- EMin = %.3f Emax = %.3f \n",InitialRigidities[iR], host_Rmax[0]);
                }

                // ->then finalize on CPU
                for (int itemp=1; itemp<prop_launch_param.blocks; itemp++) {
                    if (host_Rmax[0] < host_Rmax[itemp]) {
                        host_Rmax[0] = host_Rmax[itemp];
                    }
                }

                if (host_Rmax[0]<SimParameters.Tcentr[iR]){
                    printf("PROBLEMA: the max exiting energy is bigger than initial one\n");
                    continue;
                }
            #endif

            cudaFree(dev_maxs);

            if (VERBOSE){
                HANDLE_ERROR( cudaEventRecord( Cycle_step1, 0 ) );
                HANDLE_ERROR( cudaEventSynchronize( Cycle_step1 ) );
            }

            if (FINALSAVE) {    
                // host final states for specific energy
                struct QuasiParticle_t host_final_QuasiParts = InitQuasiPart_mem(prop_launch_param.Npart, 0, VERBOSE_2);

                // copy device final propagation states to host quasi particle states
                HANDLE_ERROR(cudaMemcpy(host_final_QuasiParts.r, dev_QuasiParts.r, prop_launch_param.Npart*sizeof(float), cudaMemcpyDeviceToHost));
                HANDLE_ERROR(cudaMemcpy(host_final_QuasiParts.th, dev_QuasiParts.th, prop_launch_param.Npart*sizeof(float), cudaMemcpyDeviceToHost));
                HANDLE_ERROR(cudaMemcpy(host_final_QuasiParts.phi, dev_QuasiParts.phi, prop_launch_param.Npart*sizeof(float), cudaMemcpyDeviceToHost));
                HANDLE_ERROR(cudaMemcpy(host_final_QuasiParts.R, dev_QuasiParts.R, prop_launch_param.Npart*sizeof(float), cudaMemcpyDeviceToHost));
                HANDLE_ERROR(cudaMemcpy(host_final_QuasiParts.t_fly, dev_QuasiParts.t_fly, prop_launch_param.Npart*sizeof(float), cudaMemcpyDeviceToHost));
                // HANDLE_ERROR(cudaMemcpy(host_final_QuasiParts.alphapath, dev_QuasiParts.alphapath, prop_launch_param.Npart*sizeof(float), cudaMemcpyDeviceToHost));

                // Saving the propagation output into a txt file
                SaveTxt_part(final_filename, prop_launch_param.Npart, host_final_QuasiParts, host_Rmax[0], VERBOSE_2);

                // Free the host particle variable for the energy on which the cycle is running
                free(host_final_QuasiParts.r);
                free(host_final_QuasiParts.th);
                free(host_final_QuasiParts.phi);
                free(host_final_QuasiParts.R);
                free(host_final_QuasiParts.t_fly);
                // free(host_final_QuasiParts.alphapath);
            } 
            
            if (VERBOSE){
                HANDLE_ERROR( cudaEventRecord( FinalSave, 0 ) );
                HANDLE_ERROR( cudaEventSynchronize( FinalSave ) );
            }

            // Definition of histogram binning as a fraction of the bin border (DeltaT=T*RelativeBinAmplitude)
            float DeltaLogR= log10(1.+Rigidity(RelativeBinAmplitude, pt));
            float LogBin0_lowEdge = log10(InitialRigidities[iR])-(DeltaLogR/2.);
            float Bin0_lowEdge = pow(10, LogBin0_lowEdge );                     // first LowEdge Bin

            Results[iR].Nbins           = ceilf(log10(host_Rmax[0]/Bin0_lowEdge)/DeltaLogR);
            Results[iR].LogBin0_lowEdge = LogBin0_lowEdge;
            Results[iR].DeltaLogR       = DeltaLogR;

            free(host_Rmax);

            // .. save to histogram ..........................................
            // Partial block histogram allocation
            float* dev_partialHistos;
            HANDLE_ERROR(cudaMalloc((void**) &dev_partialHistos, Results[iR].Nbins*prop_launch_param.blocks*sizeof(float)));

            // Final merged histogram allocation
            Results[iR].BoundaryDistribution = (float*)malloc(Results[iR].Nbins*sizeof(float));
            float* dev_Histo;
            HANDLE_ERROR(cudaMalloc((void **)&dev_Histo, Results[iR].Nbins*sizeof(float)));

            #if NVIDIA_HIST
                
                // Partial histogram atomoic sum (exploiting the shared memory)
                Rhistogram_atomic<<<prop_launch_param.blocks, prop_launch_param.threads, Results[iR].Nbins*sizeof(int)>>>(dev_QuasiParts.R, LogBin0_lowEdge, DeltaLogR , Results[iR].Nbins, prop_launch_param.Npart,  dev_partialHistos);
                
                /* int* host_partialHistos = (int*)malloc(Results[iR].Nbins*prop_launch_param.blocks*sizeof(int));
                cudaMemcpy(host_partialHistos, dev_partialHistos, Results[iR].Nbins*prop_launch_param.blocks*sizeof(int), cudaMemcpyDeviceToHost);
                printf("dev_partialHistos: \n");
                for (int i=0; i<Results[iR].Nbins*prop_launch_param.blocks; i++) {
                    printf("%d ", host_partialHistos[i]);
                } */

                cudaDeviceSynchronize();
                
                // Merge of the partial histograms and copy to the host
                TotalHisto<<<Results[iR].Nbins, prop_launch_param.blocks/2, (prop_launch_param.blocks/2)*sizeof(int)>>>(dev_partialHistos, Results[iR].Nbins, prop_launch_param.blocks, dev_Histo);
                cudaMemcpy(Results[iR].BoundaryDistribution, dev_Histo, Results[iR].Nbins*sizeof(float),cudaMemcpyDeviceToHost);

            #else

                int *dev_Nfailed;
                HANDLE_ERROR(cudaMalloc((void **) &dev_Nfailed, sizeof(int))) ;
                cudaMemset(dev_Nfailed, 0, sizeof(int));
                
                // Partial histogram atomoic sum on GPU
                histogram_atomic<<<prop_launch_param.blocks, prop_launch_param.threads>>>(dev_QuasiParts.R,  LogBin0_lowEdge, DeltaLogR, Results[iR].Nbins, prop_launch_param.Npart,  dev_partialHistos, dev_Nfailed);
                
                // Failed quasi-particle propagation count
                int Nfailed=0;
                cudaMemcpy(&Nfailed, dev_Nfailed, sizeof(int),cudaMemcpyDeviceToHost);
                Results[iR].Nregistered = prop_launch_param.Npart-Nfailed;
                
                if (VERBOSE){
                    fprintf(stdout,"-- Eventi computati : %lu \n", prop_launch_param.Npart);
                    fprintf(stdout,"-- Eventi falliti   : %d \n", Nfailed);
                    fprintf(stdout,"-- Eventi registrati: %lu \n", Results[iR].Nregistered);
                }
                cudaDeviceSynchronize();

                int histo_Nblocchi = ceil_int(Results[iR].Nbins, prop_launch_param.threads);
                
                // Merge of the partial histograms and copy to the host
                histogram_accum<<<histo_Nblocchi, prop_launch_param.threads>>>(dev_partialHistos, Results[iR].Nbins, prop_launch_param.blocks, dev_Histo);
                cudaMemcpy(Results[iR].BoundaryDistribution, dev_Histo, Results[iR].Nbins*sizeof(float),cudaMemcpyDeviceToHost);

                cudaFree(dev_Nfailed);
            #endif

            cudaFree(dev_partialHistos);
            cudaFree(dev_Histo);


            // ANNOTATION THE ONLY MEMCOPY NEEDED FROM DEVICE TO HOST ARE THE FINAL RESULTS (ALIAS THE ENERGY FINAL HISTOGRAM AND PARTICLE EXIT RESULTS)

            // .. ............................................................
            if (VERBOSE){
                HANDLE_ERROR( cudaEventRecord( Cycle_step2, 0 ) );
                HANDLE_ERROR( cudaEventSynchronize( Cycle_step2 ) );
                float   Enl00,Enl0,Enl1,Enl2, EnlIn, EnlFin;
                HANDLE_ERROR( cudaEventElapsedTime( &Enl00,
                                                    Cycle_start, Cycle_step00 ) );
                HANDLE_ERROR( cudaEventElapsedTime( &EnlIn,
                                                    Cycle_step00, InitialSave ) );
                HANDLE_ERROR( cudaEventElapsedTime( &Enl0,
                                                    InitialSave, Cycle_step0 ) );
                HANDLE_ERROR( cudaEventElapsedTime( &Enl1,
                                                    Cycle_step0, Cycle_step1 ) );
                HANDLE_ERROR( cudaEventElapsedTime( &EnlFin,
                                                    Cycle_step1, FinalSave ) );
                HANDLE_ERROR( cudaEventElapsedTime( &Enl2,
                                                    FinalSave, Cycle_step2 ) );
                printf( "-- Init              :  %3.2f ms \n", Enl00 );                                         
                printf( "-- Save initial state:  %3.2f ms \n", EnlIn );                                         
                printf( "-- Propagation phase :  %3.2f ms \n", Enl0 );
                printf( "-- Find Max          :  %3.2f ms \n", Enl1 );
                printf( "-- Save final state  :  %3.2f ms \n", EnlFin );
                printf( "-- Binning           :  %3.2f ms \n", Enl2 );    
                HANDLE_ERROR( cudaEventDestroy( Cycle_start ) ); 
                HANDLE_ERROR( cudaEventDestroy( Cycle_step00 ) );
                HANDLE_ERROR( cudaEventDestroy( InitialSave ) );
                HANDLE_ERROR( cudaEventDestroy( Cycle_step0 ) );
                HANDLE_ERROR( cudaEventDestroy( Cycle_step1 ) );
                HANDLE_ERROR( cudaEventDestroy( FinalSave ) );
                HANDLE_ERROR( cudaEventDestroy( Cycle_step2 ) );
            }
        }
        // end of the cycle on the rigidities

        if (VERBOSE){
            HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
            HANDLE_ERROR( cudaEventSynchronize( stop ) );
        }
        // Execution Time
        if (VERBOSE){
            float   elapsedTime,firstStep,memset;
            HANDLE_ERROR( cudaEventElapsedTime( &memset,
                                                    start, MemorySet ) );
            HANDLE_ERROR( cudaEventElapsedTime( &firstStep,
                                                    start, Randomstep ) );
            HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                                    start, stop ) );  
            printf( "Time to Set Memory:  %3.1f ms \n", memset );
            printf( "Time to create Rnd:  %3.1f ms (delta = %3.1f)\n", firstStep, firstStep-memset );                                     
            printf( "Time to execute   :  %3.1f ms (delta = %3.1f)\n", elapsedTime, elapsedTime-firstStep);
    
        }
      

        ////////////////////////////////////////////////////////////////
        //..... Exit results saving   ..................................
        ////////////////////////////////////////////////////////////////

        //  Save the summary histogram
        //  Free the dynamic memory

        // Save the rigidity histograms to txt file
        for (int iR=0; iR<NInitRig; iR++) {
            SaveTxt_histo(histo_filename, Results[iR].Nbins, Results[iR], VERBOSE_2);
        }

        /* save results to file .dat */
        #if HELMOD_LOAD
            FILE * pFile_Matrix=NULL;
            char RAWMatrix_name[MaxCharinFileName];
            sprintf(RAWMatrix_name,"%s_matrix_%lu.dat", SimParameters.output_file_name, (unsigned long int)getpid());

            if (VERBOSE) fprintf(stdout,"Writing Output File: %s \n", RAWMatrix_name);
            pFile_Matrix = fopen (RAWMatrix_name, "w");
            
            if (pFile_Matrix==NULL) {
                fprintf(stderr, ERR_NoOutputFile);
                fprintf(stderr, "Writing to StandardOutput instead\n");
                pFile_Matrix = stdout;
            }

            fprintf(pFile_Matrix, "# COSMICA \n");
            if (VERBOSE) fprintf(pFile_Matrix, "# Number of Input energies;\n");
            fprintf(pFile_Matrix, "%d \n", SimParameters.NT);

            for (int itemp=0; itemp<SimParameters.NT; itemp++) {
                if (VERBOSE) {
                    fprintf(pFile_Matrix,"######  Bin %d \n", itemp);
                    fprintf(pFile_Matrix,"# Rgen, Npart Gen., Npart Registered, Nbin output, log10(lower edge bin 0), Bin amplitude (in log scale)\n");
                }
                
                fprintf(pFile_Matrix,"%f %lu %lu %d %f %f \n",SimParameters.Tcentr[itemp],
                                                            SimParameters.Npart,
                                                                          Results[itemp].Nregistered,
                                                                          Results[itemp].Nbins,
                                                                          Results[itemp].LogBin0_lowEdge,
                                                                          Results[itemp].DeltaLogR);                   
                if (VERBOSE) fprintf(pFile_Matrix, "# output distribution \n");
        
                for (int itNB=0; itNB<Results[itemp].Nbins; itNB++) {
                    fprintf(pFile_Matrix, "%e ", Results[itemp].BoundaryDistribution[itNB]);
                }

                fprintf(pFile_Matrix,"\n");
                fprintf(pFile_Matrix,"#\n"); // <--- dummy line to separate results
            }

            fflush(pFile_Matrix);
            fclose(pFile_Matrix);
        #endif

        // Free the host and device memory
        cudaFree(dev_PeriodIndexes);

        free(host_QuasiParts.r);
        free(host_QuasiParts.th);
        free(host_QuasiParts.phi);
        free(host_QuasiParts.R);
        free(host_QuasiParts.t_fly);
        // free(host_QuasiParts.alphapath);

        cudaFree(dev_RndStates);
        cudaFree(dev_QuasiParts.r);
        cudaFree(dev_QuasiParts.th);
        cudaFree(dev_QuasiParts.phi);
        cudaFree(dev_QuasiParts.R);
        cudaFree(dev_QuasiParts.t_fly);
        // cudaFree(dev_QuasiParts.alphapath);

        free(host_PeriodIndexes);

        if (VERBOSE) {
            HANDLE_ERROR( cudaEventDestroy( start ) );
            HANDLE_ERROR( cudaEventDestroy( Randomstep ) );
            HANDLE_ERROR( cudaEventDestroy( stop ) );
        }      
    }
    // end of the multiple CPU thread pragma

    // Free of the initial simulation variables
    free(InitialPositions.r);
    free(InitialPositions.th);
    free(InitialPositions.phi);
    free(InitialRigidities);


    free(GPUs_profile);

    if (VERBOSE) {
        // -- Save end time of simulation into log file
        time_t tim =time(NULL);
        struct tm *local = localtime(&tim);
        printf("\nSimulation end at: %s  \n",asctime(local));
    }


    return EXIT_SUCCESS;

}