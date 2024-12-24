/**
 * \date                begin of the project Jan 2022
 * \author              Stefano Della Torre, INFN - Milano-Bicocca.
 * \mail                Stefano.dellatorre@mib.infn.it
 * \description         Particle propagation in the Heliosphere.
 * version X.Y X=major changes
 *             Y=bugfixes on major change
 */
#define VERSION "3.1"
#define MAINCU
// -----------------------------------------------------------------
// ------------------  Libraries -----------------------------------
// -----------------------------------------------------------------
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
//#include <math.h>
// .. CUDA specific
#include <curand.h>         // CUDA random number host library
#include <curand_kernel.h>  // CUDA random number device library

// .. project specific
#include "globals.h"
#include "HeliosphereModel.h"
#include "SDE.h"
#include "DiffusionModel.h"
#include "DiffusionMatrix.h"
#include "MagneticDrift.h"

/*to be cleaned*/
//#include <cuda.h>         
#include "SolarWind.h"
// -----------------------------------------------------------------
// ------------------  Defines -------------------------------------
// -----------------------------------------------------------------
// .. 
#define WELCOME "Welcome to HelMod-4-CUDA %s\n"
#define DEFAULT_PROGNAME "HelMod-4-CUDA"
#define OPTSTR "vi:h"
#define USAGE_MESSAGE "Thanks for using HelMod-4-CUDA, the usage of this program is: "
#define USAGE_FMT  "%s [-v] [-i inputfile] [-h] \n"
#define ERR_Load_Configuration_File "Error while loading simulation parameters \n"
#define LOAD_CONF_FILE_SiFile "Configuration file loaded \n"
#define LOAD_CONF_FILE_NoFile "No configuration file Specified. default value used instead \n"
#define ERR_NoOutputFile "ERROR: output file cannot be open, do you have writing permission?\n"



#define ReadingStringLenght 2000        // max lenght of each row while reading input file
// cuda - number block,thread, wrap,...
#define tpb 1024
#define hNt 128

#ifndef SetWarpPerBlock
  #define SetWarpPerBlock 32                                          // number of warp so be submitted -- modify this value to find the best performance
#endif

#ifndef MaxValueTimeStep
  #define MaxValueTimeStep 50                                         // max allowed value of time step
#endif
#ifndef MinValueTimeStep
  #define MinValueTimeStep 0.01                                       // min allowed value of time step
#endif


#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// -----------------------------------------------------------------
// ------------------  typedefs ------------------------------------
// -----------------------------------------------------------------
typedef struct {
  unsigned char verbose;                                            /* there are 3+ level of verbose. 
                                                                     *     low: main steps and essential infos
                                                                     *     med: details of all variables and steps
                                                                     *     hig: all steps info (debugger like)
                                                                     *     +: crazy request.. for hard debug!!
                                                                     */
  FILE         *input;
} options_t;

typedef struct {                                                    // Place here all simulation variables
    char  output_file_name[struct_string_lengh]="SimTest";
    unsigned long      Npart=5000;                                  // number of event to be simulated
    unsigned short      NT;                                          // number of bins of energies to be simulated
    unsigned short      NInitialPositions=0;                         // number of initial positions -> this number represent also the number of Carrington rotation that                 
    float              *Tcentr;                                     // array of energies to be simulated
    vect3D_t           *InitialPosition;                            // initial position
    PartDescription_t  IonToBeSimulated;                            // Ion to be simulated
    MonteCarloResult_t *Results;                                    // output of the code
    float RelativeBinAmplitude = 0.00855 ;                          // relative (respect 1.) amplitude of Energy bin used as X axis in BoundaryDistribution  --> delta T = T*RelativeBinAmplitude
    SimulatedHeliosphere_t HeliosphereToBeSimulated;                // Heliosphere properties for the simulation
    HeliosphereZoneProperties_t prop_medium[NMaxRegions];           // PROPerties of the interplanetary MEDIUM - Heliospheric Parameters in each Heliospheric Zone
    HeliosheatProperties_t prop_Heliosheat[NMaxRegions];            // Properties of Heliosheat
} SimParameters_t;

typedef struct{                                                     // parameters for the propagation Kernel
  unsigned long NpartPerKernelExecution;   //   32 bit              // numeber of evet to be executed by the kernel
//  particle_t    Quasi_Particle_Object;     // 6*32 bit              // initial position
//  unsigned char PeriodIndex;                                        // index of the parameter region to be used as close to Earth (heliospheric region 0)
} PropagationParameters_t; // usage 32 bit

typedef struct{
  float k0        =0 ;
  float ssn       =0 ;
  float V0        =0 ;
  float TiltAngle =0 ;
  float SmoothTilt=0 ;
  float BEarth    =0 ;
  int Polarity    =0 ;
  int SolarPhase  =0 ;
  float NMCR      =0 ;
  float Rts_nose  =0 ;
  float Rts_tail  =0 ;
  float Rhp_nose  =0 ;
  float Rhp_tail  =0 ;
} InputHeliosphericParameters_t;

typedef struct{
  float k0        =0 ;
  float V0        =0 ; // solar wind at termination shock
} InputHeliosheatParameters_t;


// -----------------------------------------------------------------
// ------------------  External Declaration  -----------------------
// -----------------------------------------------------------------
extern int errno;
extern char *optarg;
extern int opterr, optind;




// -----------------------------------------------------------------
// ------------------  Global Variables declaration ----------------
// -----------------------------------------------------------------
/* none */                                               

// -----------------------------------------------------------------
// ------------  Device Constant Variables declaration -------------
// -----------------------------------------------------------------
__constant__ SimulatedHeliosphere_t      Heliosphere;            // Heliosphere properties include Local Interplanetary medium parameters
__constant__ HeliosphereZoneProperties_t LIM[NMaxRegions];       // inner heliosphere
__constant__ HeliosheatProperties_t      HS[NMaxRegions];        // heliosheat
// -----------------------------------------------------------------
// ------------------  function Prototypes -------------------------
// -----------------------------------------------------------------
void usage(char *, int );
int Load_Configuration_File(options_t*, SimParameters_t &);            
static void HandleError( cudaError_t , const char *, int) ;
int PrintError (char*, char *, int );


int ceil_int(int,int);
int floor_int(int, int);
__global__ void kernel_max(particle_t *, float *, unsigned long);

__global__ void HeliosphericPropagation(curandStatePhilox4_32_10_t *, PropagationParameters_t,particle_t *, int *) ;
__global__ void histogram_atomic(const particle_t *, const float , const float  , const int , const unsigned long , float * , int *);
__global__ void histogram_accum(const float *,  const int , const int , float *);
__global__ void init_rdmgenerator(curandStatePhilox4_32_10_t *, unsigned long long );

// -----------------------------------------------------------------
// ------------------  Main Code -----------------------------------
// -----------------------------------------------------------------
int main( int argc, char* argv[] ) {

  ////////////////////////////////////////////////////////////////
  //..... Initialize program   ...................................
  //  initialize variables from input file or default value
  ////////////////////////////////////////////////////////////////

  //.................. load arguments from command line ..........
  int opt;                                                          
  options_t options = { false, stdin };
  opterr = 0;
  while ((opt = getopt(argc, argv, OPTSTR)) != EOF)
    switch(opt) {
      case 'i':
        if (!(options.input = fopen(optarg, "r")) ){
          perror(optarg);
          exit(EXIT_FAILURE);
          /* NOTREACHED */
        }
        break;

      case 'v':
        options.verbose += 1;
        break;

      case 'h':
      default:
        usage(basename(argv[0]), opt);
        /* NOTREACHED */
        break;
    }
  if (options.verbose) { 
    printf(WELCOME,VERSION);
    switch (options.verbose){
      case VERBOSE_low: 
        printf("Verbose level: low\n");
        break;
      case VERBOSE_med: 
        printf("Verbose level: medium\n");
        break;
      case VERBOSE_hig: 
        printf("Verbose level: high\n");
        break;
      default:
        printf("Verbose level: crazy\n");
        break;
    }
    
    if (options.verbose>=VERBOSE_med){
      fprintf(stderr,"-- --- Init ---\n");
      fprintf(stderr,"-- you entered %d arguments:\n",argc);
      for (int i = 0; i<argc; i++){ fprintf(stderr,"-->  %s \n",argv[i]);}
    }
  }
  if (options.verbose)
  {
    // -- Save initial time of simulation into log file
    time_t tim =time(NULL);
    struct tm *local = localtime(&tim);
    printf("\nSimulation started at: %s  \n",asctime(local));
  }
  //................. load simulation parameters 
  SimParameters_t SimParameters;
  if (Load_Configuration_File(&options,SimParameters) != EXIT_SUCCESS) {
    perror(ERR_Load_Configuration_File);
    exit(EXIT_FAILURE);
    /* NOTREACHED */
  }





  ////////////////////////////////////////////////////////////////
  //..... Initialize CUDA and GPU infos   ........................
  ////////////////////////////////////////////////////////////////


  // retrive some information from the GPU for info
  cudaDeviceProp  prop;
  int N_GPU_count;
  HANDLE_ERROR( cudaGetDeviceCount( &N_GPU_count ) );
  if (N_GPU_count < 1) {
    fprintf(stderr,"no CUDA capable devices were detected\n");
    exit(EXIT_FAILURE);
  }
  // .. for debugging
  if (options.verbose){
    printf( "----- CPU infos -----\n");
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf( "----- GPU infos -----\n" );
    printf( "There are %d CUDA enabled devices \n",N_GPU_count );
    if (options.verbose>=VERBOSE_med){
      for (int i=0; i< N_GPU_count; i++) {
          HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
          printf( "--   --- General Information for device %d ---\n", i );
          printf( "-- Name:  %s\n", prop.name );
          printf( "-- Compute capability:  %d.%d\n", prop.major, prop.minor );
          printf( "-- Clock rate:  %d\n", prop.clockRate );
          printf( "-- Device copy overlap:  " );
          if (prop.deviceOverlap)
              printf( "Enabled\n" );
          else
              printf( "Disabled\n");
          printf( "-- Kernel execution timeout :  " );
          if (prop.kernelExecTimeoutEnabled)
              printf( "Enabled\n" );
          else
              printf( "Disabled\n" );

          printf( "--    --- Memory Information for device %d ---\n", i );
          printf( "-- Total global mem:  %ld\n", prop.totalGlobalMem );
          printf( "-- Total constant Mem:  %ld\n", prop.totalConstMem );
          printf( "-- Max mem pitch:  %ld\n", prop.memPitch );
          printf( "-- Texture Alignment:  %ld\n", prop.textureAlignment );

          printf( "--    --- MP Information for device %d ---\n", i );
          printf( "-- Multiprocessor count:  %d\n",
                      prop.multiProcessorCount );
          printf( "-- Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
          printf( "-- Registers per mp:  %d\n", prop.regsPerBlock );
          printf( "-- Threads in warp:  %d\n", prop.warpSize );
          printf( "-- Max threads per block:  %d\n",
                      prop.maxThreadsPerBlock );
          printf( "-- Max thread dimensions:  (%d, %d, %d)\n",
                      prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                      prop.maxThreadsDim[2] );
          printf( "-- Max grid dimensions:  (%d, %d, %d)\n",
                      prop.maxGridSize[0], prop.maxGridSize[1],
                      prop.maxGridSize[2] );
          printf( "-- \n" );
      }
    }
    
  }

  ////////////////////////////////////////////////////////////////
  //..... Rescale Heliosphere to an effective one  ...............
  ////////////////////////////////////////////////////////////////
  for (int ipos=0; ipos<SimParameters.NInitialPositions;ipos++){
    SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos]=SimParameters.HeliosphereToBeSimulated.RadBoundary_real[ipos];
    RescaleToEffectiveHeliosphere(SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos],SimParameters.InitialPosition[ipos]);
    if (options.verbose>=VERBOSE_med){
      fprintf(stderr,"--- Zone %d \n",ipos);
      fprintf(stderr,"--- !! Effective Heliosphere --> effective boundaries: TS_nose=%f TS_tail=%f Rhp_nose=%f  Rhp_tail=%f \n",SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos].Rts_nose,SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos].Rts_tail,SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos].Rhp_nose,SimParameters.HeliosphereToBeSimulated.RadBoundary_effe[ipos].Rhp_tail);
      fprintf(stderr,"--- !! Effective Heliosphere --> new Source Position: r=%f th=%f phi=%f \n",SimParameters.InitialPosition[ipos].r,SimParameters.InitialPosition[ipos].th,SimParameters.InitialPosition[ipos].phi);
    }
  }

  ////////////////////////////////////////////////////////////////
  //..... Initialize Npart assuming gpu 0   ......................
  ////////////////////////////////////////////////////////////////

  // .. the Kernel executed all particles, each on with one GPU-thread
  //    to optimize the use of the GPU, this number should be a multiple of prop.warpSize
  //    moreover the same kernel deal with different initial positions, so that the number of particle,
  //    i.e. number of threads, should be a multiple of  prop.warpSize*SimParameters.NInitialPositions
  //    in this way each warp process particle with the same initial position and period
  HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0 ) ); // using GPU0 as reference for all --> this should be modified if GPU architecture are different
  SimParameters.Npart = ceil_int(SimParameters.Npart,(prop.warpSize*SimParameters.NInitialPositions))*(prop.warpSize*SimParameters.NInitialPositions);
  
  if (options.verbose)
  {
    printf("-- The number of particles is rounded to %lu fit the warpsize\n",SimParameters.Npart);
  }  

  ////////////////////////////////////////////////////////////////
  //..... Initialize CPU threads   ...............................
  ////////////////////////////////////////////////////////////////
  // run as many CPU threads as there are CUDA devices
  //   each CPU thread controls a different device, processing its
  //   portion of the data. 
  omp_set_num_threads(N_GPU_count);  // create as many CPU threads as there are CUDA devices
  // start cpu threads
#pragma omp parallel
  {
    unsigned int cpu_thread_id = omp_get_thread_num();     // identificativo del CPU-thread
    unsigned int num_cpu_threads = omp_get_num_threads();  // numero totale di CPU-thread
    int gpu_id = cpu_thread_id % N_GPU_count;              // seleziona la id della GPU da usare. "% num_gpus" allows more CPU threads than GPU devices
    HANDLE_ERROR( cudaSetDevice(gpu_id));                  // seleziona la GPU
    if (options.verbose)
    { 
      HANDLE_ERROR(cudaGetDevice(&gpu_id));
      printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);
    }
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, gpu_id ) );

    // .. defining parameters that are common to all GPU-thread
    PropagationParameters_t HeliosphericPropagation_Param;
    HeliosphericPropagation_Param.NpartPerKernelExecution = SimParameters.Npart;
    ///////////////////////////////////////////////////////////////////////////
    /////////////////// Block and Thread Counts ///////////////////////////////

    
     
    // .. evaluating the number of block and threads needed to evalute all Npart in the kernel
    //    the max number of warp is passed by compiling variable _SetWarpPerBlock_ 
    //    the actual number of warp needed is given by NpartPerKernelExecution/warpSize
    //    an internal check should control that the numbero of warp required fulfill the 
    //    tech. spec. of the board

    if (SetWarpPerBlock>prop.maxThreadsPerBlock/prop.warpSize)
    {
      fprintf(stderr,"ERROR:: the setted number SetWarpPerBlock (defined while compiling) exceded the max number of warp in a block for this card (%d).\n",prop.maxThreadsPerBlock/prop.warpSize);
      exit(EXIT_FAILURE);
    }

    // Block and Thread count for Propagation kernel
    //     Number of block = NThread/MaxNumberOfThreadInTheBlock = ceil ( Npart / (SetWarpPerBlock*warpSize) )
    //     Number of threadPerBlock = ceil(NThread/Number of block)
    unsigned int blockCount      = (unsigned int)ceil_int(HeliosphericPropagation_Param.NpartPerKernelExecution,SetWarpPerBlock*prop.warpSize); 
    unsigned int threadsPerBlock = (unsigned int)ceil_int(HeliosphericPropagation_Param.NpartPerKernelExecution,blockCount);
    if ( threadsPerBlock>prop.maxThreadsPerBlock || blockCount>prop.maxGridSize[0])
    {
      fprintf(stderr,"------- propagation Kernel -----------------\n");
      fprintf(stderr,"ERROR:: Number of Threads per block or number of blocks not allowed for this device\n");
      fprintf(stderr,"        Number of Threads per Block setted %d - max allowed %d\n",threadsPerBlock,prop.maxThreadsPerBlock);
      fprintf(stderr,"        Number of Blocks setted %d - max allowed %d\n",blockCount,prop.maxGridSize[0]);
      exit(EXIT_FAILURE);
    }
    if (options.verbose>=VERBOSE_med)
    {
      printf("------- propagation Kernel -----------------\n");
      printf("-- Max Number of Warp in a Block   : %d \n",SetWarpPerBlock);
      printf("-- Number of blocks                : %d \n",blockCount);
      printf("-- Number of threadsPerBlock       : %d \n",threadsPerBlock);
    } 


    // Block and Thread count for atomic Histogram Kernel
    unsigned int histo_blockCount      = (unsigned int)ceil_int(SimParameters.Npart,prop.maxThreadsPerBlock); 
    unsigned int histo_threadsPerBlock = (unsigned int)ceil_int(SimParameters.Npart,histo_blockCount);
    if ( histo_threadsPerBlock>prop.maxThreadsPerBlock || histo_blockCount>prop.maxGridSize[0])
    {
      fprintf(stderr,"------- Histogram Kernel -----------------\n");
      fprintf(stderr,"ERROR:: Number of Threads per block or number of blocks not allowed for this device\n");
      fprintf(stderr,"        Number of Threads per Block setted %d - max allowed %d\n",histo_threadsPerBlock,prop.maxThreadsPerBlock);
      fprintf(stderr,"        Number of Blocks setted %d - max allowed %d\n",histo_blockCount,prop.maxGridSize[0]);
      exit(EXIT_FAILURE);
    }
    if (options.verbose>=VERBOSE_med)
    {
      printf("------- Histogram Kernel -----------------\n");
      printf("-- Max Number of Warp in a Block   : %d \n",prop.maxThreadsPerBlock);
      printf("-- Number of blocks                : %d \n",histo_blockCount);
      printf("-- Number of threadsPerBlock       : %d \n",histo_threadsPerBlock);
    } 

    /////////////////// end Block and Thread Counts ///////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    

    // .. capture the start time
    cudaEvent_t     start,MemorySet,Randomstep, stop;
    cudaEvent_t     Cycle_start,Cycle_step00,Cycle_step0,Cycle_step1,Cycle_step2;
    if (options.verbose){
      HANDLE_ERROR( cudaEventCreate( &start ) );
      HANDLE_ERROR( cudaEventCreate( &MemorySet ) );
      HANDLE_ERROR( cudaEventCreate( &Randomstep ) );
      HANDLE_ERROR( cudaEventCreate( &stop ) );
      HANDLE_ERROR( cudaEventRecord( start, 0 ) );
    }


    ///////////////////////////////////////////////////////////////////////////
    // .. Device Constant Memory
    cudaMemcpyToSymbol(Heliosphere, &SimParameters.HeliosphereToBeSimulated, sizeof(SimulatedHeliosphere_t));
    cudaMemcpyToSymbol(LIM, &SimParameters.prop_medium   , NMaxRegions*sizeof(HeliosphereZoneProperties_t));
    cudaMemcpyToSymbol(HS, &SimParameters.prop_Heliosheat, NMaxRegions*sizeof(HeliosheatProperties_t));

    ///////////////////////////////////////////////////////////////////////////
    // .. in device global memory there will be two arrays
    //    One is the Final state of the propagation that should be reinizialized for each kernel
    //    with the initial position (the same variable is used for initial and final state, this because
    //    for each new kernel the initial energy change, thus a new copyToDevice should be done)
    //    The second is the array that assign each particle a specific PeriodIndex to be used to
    //    load the correct propagation parameters runtime
  
    particle_t *dev_FinalPropState;                     // device input/output of propagation kernel
    particle_t *host_InitialPropState;                  // host initial state of propagation kernel
    int *dev_PeriodIndexes;                             // device PeriodIndex array
    int *host_PeriodIndexes;                            // host PeriodIndex array
    // allocate on device
    HANDLE_ERROR( cudaMalloc( (void**)&dev_FinalPropState, SimParameters.Npart * sizeof( particle_t ) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_PeriodIndexes , SimParameters.Npart * sizeof( int ) ) );
    cudaDeviceSynchronize();
    // allocate on host 
    host_InitialPropState = (particle_t*)malloc( SimParameters.Npart * sizeof( particle_t ) );
    host_PeriodIndexes    = (int*)malloc( SimParameters.Npart * sizeof( int ) ); 
    // initialize the host array
    // .. every SimParameters.Npart/SimParameters.NInitialPositions elements, should be filled with
    //    initial position and period index from 0 to NInitialPositions-1
    for(int iPart=0; iPart<SimParameters.Npart; iPart++)
    {
      int PeriodIndex=  floor_int(iPart*SimParameters.NInitialPositions,SimParameters.Npart );
      host_PeriodIndexes[iPart]=PeriodIndex;
      host_InitialPropState[iPart].part.r  = SimParameters.InitialPosition[PeriodIndex].r;
      host_InitialPropState[iPart].part.th = SimParameters.InitialPosition[PeriodIndex].th;
      host_InitialPropState[iPart].part.phi= SimParameters.InitialPosition[PeriodIndex].phi;
      host_InitialPropState[iPart].pt      = SimParameters.IonToBeSimulated ;
      host_InitialPropState[iPart].alphapath = 0;
      host_InitialPropState[iPart].prop_time = 0; 
    }
    if (options.verbose>=VERBOSE_hig)
    {
      printf("### ---- initialize host_InitialPropState --------------\n");
      for(int iPart=0; iPart<SimParameters.Npart; iPart++)
      {
        printf("### Particle %d: PeriodIndex=%d position (%e,%e,%e) Ion Z:%.0f A:%.0f T0:%.0f\n",iPart,
                                                                       host_PeriodIndexes[iPart],
                                                                       host_InitialPropState[iPart].part.r,
                                                                       host_InitialPropState[iPart].part.th,
                                                                       host_InitialPropState[iPart].part.phi,
                                                                       host_InitialPropState[iPart].pt.Z,
                                                                       host_InitialPropState[iPart].pt.A,
                                                                       host_InitialPropState[iPart].pt.T0);
      }
    }

    // copy host_PeriodIndexes to dev_PeriodIndexes and free memory
    HANDLE_ERROR(cudaMemcpy(dev_PeriodIndexes, host_PeriodIndexes, SimParameters.Npart * sizeof( int ),cudaMemcpyHostToDevice));
    free(host_PeriodIndexes);

    if (options.verbose){
      HANDLE_ERROR( cudaEventRecord( MemorySet, 0 ) );
      HANDLE_ERROR( cudaEventSynchronize( MemorySet ) );
    }
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////
    //..... Initialize Random Number generator   ...................
    ////////////////////////////////////////////////////////////////

    // .. random generator
    curandStatePhilox4_32_10_t *devRndStates;                                                         // Definisce il tipo di generatore 
    HANDLE_ERROR(cudaMalloc((void **)&devRndStates, HeliosphericPropagation_Param.NpartPerKernelExecution*sizeof(curandStatePhilox4_32_10_t)));     
                                                                                      // alloca la memoria per un vettore di Stati in modo che ogni thread abbia il suo generatore di numeri casuali
                                                                                      // dal manuale For the highest quality parallel pseudorandom number generation, each experiment should be assigned 
                                                                                      // a unique seed value. Within an experiment, each thread of computation should be assigned a unique id number. 
                                                                                      // If an experiment spans multiple kernel launches, it is recommended that threads between kernel launches be given 
                                                                                      // the same seed, and id numbers be assigned in a monotonically increasing way. If the same configuration of threads 
                                                                                      // is launched, random state can be preserved in global memory between launches to avoid state setup time.
    unsigned long Rnd_seed=getpid()+time(NULL)+gpu_id;
    init_rdmgenerator<<<blockCount,threadsPerBlock>>>(devRndStates, Rnd_seed);
    cudaDeviceSynchronize();
    if (options.verbose){
      HANDLE_ERROR( cudaEventRecord( Randomstep, 0 ) );
      HANDLE_ERROR( cudaEventSynchronize( Randomstep ) );
      if (options.verbose>=VERBOSE_med){
        fprintf(stdout,"--- Random Generator Seed: %lu \n",Rnd_seed);
      }
    }


    ////////////////////////////////////////////////////////////////
    //..... Main cycle of particle propagation   ...................
    ////////////////////////////////////////////////////////////////
    


     
    /* cycle con particle energy and submit propagation kernels and save to istogram kernel*/
    for (int iNT=gpu_id; iNT<SimParameters.NT ; iNT+=N_GPU_count)
    {
      if (options.verbose){
        fprintf(stdout,"\n-- Cycle on Energy[%d]: %.2f \n",iNT,SimParameters.Tcentr[iNT]);
      }
      if (options.verbose){
        HANDLE_ERROR( cudaEventCreate( &Cycle_start ) );
        HANDLE_ERROR( cudaEventCreate( &Cycle_step00 ) );
        HANDLE_ERROR( cudaEventCreate( &Cycle_step0 ) );
        HANDLE_ERROR( cudaEventCreate( &Cycle_step1 ) );
        HANDLE_ERROR( cudaEventCreate( &Cycle_step2 ) );
        HANDLE_ERROR( cudaEventRecord( Cycle_start, 0 ) );
      }
      // .. init kernel parameters .....................................
      for(int iPart=0; iPart<SimParameters.Npart; iPart++)
      {
        host_InitialPropState[iPart].part.Ek  = SimParameters.Tcentr[iNT];
      }
      HANDLE_ERROR(cudaMemcpy(dev_FinalPropState, host_InitialPropState, SimParameters.Npart * sizeof( particle_t ),cudaMemcpyHostToDevice));
      if (options.verbose){
        HANDLE_ERROR( cudaEventRecord( Cycle_step00, 0 ) );
        HANDLE_ERROR( cudaEventSynchronize( Cycle_step00 ) );
      }
      
      //printf("%d %d\n",blockCount,threadsPerBlock);
      // .. propagate events ...........................................
      HeliosphericPropagation<<<blockCount,threadsPerBlock>>>(devRndStates,HeliosphericPropagation_Param,dev_FinalPropState,dev_PeriodIndexes) ;
      
      HANDLE_ERROR(cudaPeekAtLastError() );
      HANDLE_ERROR(cudaDeviceSynchronize());
      
      if (options.verbose){
        HANDLE_ERROR( cudaEventRecord( Cycle_step0, 0 ) );
        HANDLE_ERROR( cudaEventSynchronize( Cycle_step0 ) );
      }
      // --------------- DEBUG LINES -------------------
      // particle_t *a;
      // a = (particle_t*)malloc(SimParameters.Npart * sizeof(particle_t));
      // cudaMemcpy(a, dev_FinalPropState, SimParameters.Npart*sizeof(particle_t),cudaMemcpyDeviceToHost);
      // fprintf(stdout,"--- EValues values: ");
      // for (int itemp=SimParameters.Npart-4; itemp<SimParameters.Npart; itemp++){
      //     fprintf(stdout,"%.2f ",a[itemp].Ek);
      // }
      // fprintf(stdout,"\n");
      // ----------------------------------------------
      
      // .. Find Emax ..................................................
      int BlockPerFind = ceil_int(SimParameters.Npart,tpb)  ;
      if (options.verbose>=VERBOSE_med){
        fprintf(stdout,"--- BlockPerFind: %d \n",BlockPerFind);
      }
      // ->first check on GPU
      float *dev_maxvect;
      HANDLE_ERROR(cudaMalloc((void **) &dev_maxvect, BlockPerFind*sizeof(float))) ;
      kernel_max<<<BlockPerFind,tpb>>>(dev_FinalPropState, dev_maxvect, SimParameters.Npart);
      cudaDeviceSynchronize();
      float *maxvect;
      maxvect = (float*)malloc(BlockPerFind * sizeof(float));
      cudaMemcpy(maxvect, dev_maxvect, BlockPerFind*sizeof(float),cudaMemcpyDeviceToHost);
      // ->then finalize on CPU
      float BinE_Max = maxvect[0];
      for (int itemp=1; itemp<BlockPerFind; itemp++){
          if (BinE_Max<maxvect[itemp]){
            BinE_Max = maxvect[itemp];
          }
      }    

      if (options.verbose>=VERBOSE_med){
        fprintf(stdout,"--- Max values: ");
        for (int itemp=0; itemp<BlockPerFind; itemp++){
          fprintf(stdout,"%.2f ",maxvect[itemp]);
        }
        fprintf(stdout,"\n");
        fprintf(stdout,"--- EMin = %.3f Emax = %.3f \n",SimParameters.Tcentr[iNT],BinE_Max);
      }
      if (BinE_Max<SimParameters.Tcentr[iNT]){
        printf("------------------------- PROBLEMA ----------------------");
        continue;
      }
      // -> free memory on device
      free(maxvect);
      cudaFree(dev_maxvect);
      if (options.verbose){
        HANDLE_ERROR( cudaEventRecord( Cycle_step1, 0 ) );
        HANDLE_ERROR( cudaEventSynchronize( Cycle_step1 ) );
      }
      // .. define histogram binning ...................................
      float DeltaLogT = log10(1.+SimParameters.RelativeBinAmplitude);     // amplitude of binning; defined as a fraction of the bin border (DeltaT=T*RelativeBinAmplitude)
      float LogBin0_lowEdge = log10(SimParameters.Tcentr[iNT])-(DeltaLogT/2.);
      float Bin0_lowEdge = pow(10, LogBin0_lowEdge );                     // first LowEdge Bin
      SimParameters.Results[iNT].Nbins           = ceilf( log10(BinE_Max/Bin0_lowEdge) / DeltaLogT );
      SimParameters.Results[iNT].LogBin0_lowEdge = LogBin0_lowEdge;
      SimParameters.Results[iNT].DeltaLogT       = DeltaLogT;
      

      if (options.verbose>=VERBOSE_med){
        float3          *OuterEnergyBin;
        OuterEnergyBin  = (float3*)malloc( SimParameters.Results[iNT].Nbins * sizeof(float3) );
        for (int itemp=0; itemp<SimParameters.Results[iNT].Nbins; itemp++){
          float3 bintemp;
          bintemp.x = pow(10.,LogBin0_lowEdge+itemp*DeltaLogT);
          bintemp.z = pow(10.,LogBin0_lowEdge+(itemp+1)*DeltaLogT);
          bintemp.y = (bintemp.x+bintemp.z)/2.;
          OuterEnergyBin[itemp]=bintemp;
        }
        fprintf(stdout,"--- N Output binning: %d \n",SimParameters.Results[iNT].Nbins);
        float3 bintemp = OuterEnergyBin[0];
        fprintf(stdout,"--- Binning first : [%.2f,%.2f,%.2f] \n",bintemp.x,bintemp.y,bintemp.z);
        bintemp = OuterEnergyBin[SimParameters.Results[iNT].Nbins-1];
        fprintf(stdout,"--- Binning last : [%.2f,%.2f,%.2f] \n",bintemp.x,bintemp.y,bintemp.z);
        free(OuterEnergyBin);
      }    


      

      // .. save to histogram ..........................................
      SimParameters.Results[iNT].BoundaryDistribution = (float*)malloc( SimParameters.Results[iNT].Nbins * sizeof(float) );
      float *dev_partialHisto;
      HANDLE_ERROR(cudaMalloc((void **) &dev_partialHisto, SimParameters.Results[iNT].Nbins *histo_blockCount*sizeof(float))) ;
      int *dev_Nfailed;
      HANDLE_ERROR(cudaMalloc((void **) &dev_Nfailed, sizeof(int))) ;
      cudaMemset(dev_Nfailed,0,sizeof(int));
      histogram_atomic<<<histo_blockCount,histo_threadsPerBlock>>>(dev_FinalPropState, 
                                                       LogBin0_lowEdge,
                                                       DeltaLogT,
                                                       SimParameters.Results[iNT].Nbins,
                                                       SimParameters.Npart, 
                                                       dev_partialHisto,
                                                       dev_Nfailed);
      int Nfailed=0;
      cudaMemcpy(&Nfailed, dev_Nfailed, sizeof(int),cudaMemcpyDeviceToHost);
      SimParameters.Results[iNT].Nregistered=SimParameters.Npart-Nfailed;
      if (options.verbose){
        fprintf(stdout,"-- Eventi computati : %lu \n",SimParameters.Npart);
        fprintf(stdout,"-- Eventi falliti   : %d \n",Nfailed);
        fprintf(stdout,"-- Eventi registrati: %lu \n",SimParameters.Results[iNT].Nregistered);
      }
      cudaDeviceSynchronize();
      float *dev_Histo;
      HANDLE_ERROR(cudaMalloc((void **) &dev_Histo, SimParameters.Results[iNT].Nbins *sizeof(float))) ;
      int histo_Nthreads = hNt;
      int histo_Nblocchi = ceil_int(SimParameters.Results[iNT].Nbins,histo_Nthreads);
      
      histogram_accum<<<histo_Nblocchi,histo_Nthreads>>>(dev_partialHisto,  
                                                         SimParameters.Results[iNT].Nbins,
                                                         histo_blockCount,
                                                         dev_Histo);
      cudaMemcpy(SimParameters.Results[iNT].BoundaryDistribution, dev_Histo, SimParameters.Results[iNT].Nbins*sizeof(float),cudaMemcpyDeviceToHost);

      cudaFree(dev_partialHisto);
      cudaFree(dev_Histo);
      cudaFree(dev_Nfailed);


      if (options.verbose>=VERBOSE_med){
        fprintf(stdout,"--- OutpuValues: ");
        for (int itemp=SimParameters.Results[iNT].Nbins-3; itemp<SimParameters.Results[iNT].Nbins; itemp++){
          fprintf(stdout,"(%d,%.2f) ",itemp,SimParameters.Results[iNT].BoundaryDistribution[itemp]);
        }
        fprintf(stdout,"\n");
        long int C=0;
        for (int itemp=0; itemp<SimParameters.Results[iNT].Nbins; itemp++){ C+=SimParameters.Results[iNT].BoundaryDistribution[itemp];}
        fprintf(stdout,"--- Sum of the istogram: %ld \n",C);
      }
      // .. ............................................................
      if (options.verbose){
        HANDLE_ERROR( cudaEventRecord( Cycle_step2, 0 ) );
        HANDLE_ERROR( cudaEventSynchronize( Cycle_step2 ) );
        float   Enl00,Enl0,Enl1,Enl2;
        HANDLE_ERROR( cudaEventElapsedTime( &Enl0,
                                              Cycle_start, Cycle_step00 ) );
        HANDLE_ERROR( cudaEventElapsedTime( &Enl00,
                                              Cycle_step00, Cycle_step0 ) );
        HANDLE_ERROR( cudaEventElapsedTime( &Enl1,
                                              Cycle_step0, Cycle_step1 ) );
        HANDLE_ERROR( cudaEventElapsedTime( &Enl2,
                                              Cycle_step1, Cycle_step2 ) ); 
        printf( "-- Init              :  %3.2f ms \n", Enl0 );                                          
        printf( "-- propagation phase :  %3.2f ms \n", Enl00 );
        printf( "-- Find Max          :  %3.2f ms \n", Enl1 );
        printf( "-- Binning           :  %3.2f ms \n", Enl2 );    
        HANDLE_ERROR( cudaEventDestroy( Cycle_start ) ); 
        HANDLE_ERROR( cudaEventDestroy( Cycle_step00 ) );
        HANDLE_ERROR( cudaEventDestroy( Cycle_step0 ) );
        HANDLE_ERROR( cudaEventDestroy( Cycle_step1 ) );
        HANDLE_ERROR( cudaEventDestroy( Cycle_step2 ) );
      }

    } // .. end Cycle on iNT
    cudaDeviceSynchronize();
    if (options.verbose){
      HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
      HANDLE_ERROR( cudaEventSynchronize( stop ) );
    }
    // Execution Time
    if (options.verbose){
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

    // free memory
    if (options.verbose){
      HANDLE_ERROR( cudaEventDestroy( start ) );
      HANDLE_ERROR( cudaEventDestroy( Randomstep ) );
      HANDLE_ERROR( cudaEventDestroy( stop ) );
    }
    cudaFree(dev_FinalPropState);
    cudaFree(dev_PeriodIndexes);
    free(host_InitialPropState);
    cudaFree(devRndStates);


  }//end pragma
  if (cudaSuccess != cudaGetLastError())
  {
    fprintf(stderr,"Last CUDA Error: %s\n", cudaGetErrorString(cudaGetLastError()));
  }
  ////////////////////////////////////////////////////////////////
  //..... Saving results   .......................................
  ////////////////////////////////////////////////////////////////
  if (options.verbose){ fprintf(stdout,"----- Saving results -----\n");}
  /* save results to file */
  FILE * pFile_Matrix=NULL;
  char RAWMatrix_name[MaxCharinFileName];
  sprintf(RAWMatrix_name,"%s_matrix_%lu.dat",SimParameters.output_file_name,(unsigned long int)getpid());
  if (options.verbose){ fprintf(stdout,"Writing Output File: %s \n",RAWMatrix_name);}
  pFile_Matrix = fopen (RAWMatrix_name,"w");
  if (pFile_Matrix==NULL)
  {
      fprintf(stderr,ERR_NoOutputFile);
      fprintf(stderr,"Writing to StandardOutput instead\n");
      pFile_Matrix=stdout;
  }
  fprintf(pFile_Matrix,"# %s v%s\n",DEFAULT_PROGNAME,VERSION);
  if (options.verbose){ fprintf(pFile_Matrix,"# Number of Input energies;\n"); }
  fprintf(pFile_Matrix,"%d \n",SimParameters.NT);
  for (int itemp=0; itemp<SimParameters.NT; itemp++){
    if (options.verbose){
      fprintf(pFile_Matrix,"######  Bin %d \n",itemp);
      fprintf(pFile_Matrix,"# Egen, Npart Gen., Npart Registered, Nbin output, log10(lower edge bin 0), Bin amplitude (in log scale)\n");
    }
    fprintf(pFile_Matrix,"%f %lu %lu %d %f %f \n",SimParameters.Tcentr[itemp],
                                                  SimParameters.Npart,
                                                  SimParameters.Results[itemp].Nregistered,
                                                  SimParameters.Results[itemp].Nbins,
                                                  SimParameters.Results[itemp].LogBin0_lowEdge,
                                                  SimParameters.Results[itemp].DeltaLogT);                   
    if (options.verbose){
      fprintf(pFile_Matrix,"# output distribution \n");
    }   
    for (int itNB=0; itNB<SimParameters.Results[itemp].Nbins; itNB++)
    {
      fprintf(pFile_Matrix,"%e ",SimParameters.Results[itemp].BoundaryDistribution[itNB]);
    }
    fprintf(pFile_Matrix,"\n");
    fprintf(pFile_Matrix,"#\n"); // <--- dummy line to separate results
  }

  fflush(pFile_Matrix);
  fclose(pFile_Matrix);

  ////////////////////////////////////////////////////////////////
  //..... Close program   ...................................
  //  
  ////////////////////////////////////////////////////////////////

  //free memory
  free(SimParameters.Tcentr);  
  free(SimParameters.InitialPosition);
  for (int iNT=0; iNT<SimParameters.NT ; iNT++)
  { 
    free(SimParameters.Results[iNT].BoundaryDistribution);
  }
  free(SimParameters.Results);

  if (options.verbose)
  {
    // -- Save end time of simulation into log file
    time_t tim =time(NULL);
    struct tm *local = localtime(&tim);
    printf("\nSimulation end at: %s  \n",asctime(local));
  }

  return EXIT_SUCCESS;

}

// -----------------------------------------------------------------
// ------------------  Function Declatations -----------------------
// -----------------------------------------------------------------

// .. loading program options
void usage(char *progname, int opt) {
   fprintf(stderr, USAGE_MESSAGE); 
   fprintf(stderr, USAGE_FMT, progname?progname:DEFAULT_PROGNAME);
   exit(EXIT_FAILURE);
   /* NOTREACHED */
}

int PrintError (const char *var, char *value, int zone){
  fprintf (stderr,"ERROR: %s value not valid [actual value %s for region %d] \n",var,value,zone); 
  return EXIT_FAILURE; 
}

// .. split comma separated value
unsigned char SplitCSVString(const char *InputString, float **Outputarray)
{
  unsigned short Nelements=0;
  char delim[] = ",";
  char *token;
  char cp_value[ReadingStringLenght];
  strncpy(cp_value,InputString,ReadingStringLenght);  // strtok modify the original string, since wehave to use it twice, we need a copy of "value"
        
  // ......  read first time the string and count the number of energies
  int i_split=0;
  token = strtok(cp_value, delim);
  while( token != NULL ) 
  {
    token = strtok(NULL, delim);
    i_split++;
  }
  Nelements=i_split;
  //printf("%d\n",Nelements);
  // ...... Read again and save value
  *Outputarray = (float*)malloc( Nelements * sizeof(float) );
  i_split=0;
  strncpy(cp_value,InputString,ReadingStringLenght);
  token = strtok(cp_value, delim);
  while( token != NULL ) 
  {
    (*Outputarray)[i_split]= atof(token);
    token = strtok(NULL, delim);
    i_split++;
  }
  free(token);
  // for (int i=0; i<Nelements; i++)
  // {
  //   printf("%f\n",Outputarray[i]);
  // }
  return Nelements;
}

// .. Loading configuration File
int Load_Configuration_File(options_t *options, SimParameters_t &SimParameters){
  // .. check integrity
  if (!options) {
    errno = EINVAL;
    return EXIT_FAILURE;
  }

  if (!options->input) {
    errno = ENOENT;
    return EXIT_FAILURE;
  }

  // .. Use default flags
  bool UseDefault_initial_energy=true;

  // .. Heliospheric parameters
  int NumberParameterLoaded  = 0;  // count number of parameters loaded
  int NumberHeliosheatParLoaded =0 ; 
  unsigned short Nregions=0;
  InputHeliosphericParameters_t IHP[NMaxRegions]; //
  InputHeliosheatParameters_t   IHS[NMaxRegions];

  // .. initial positions
  float *r; 
  float *th;
  float *ph;
  unsigned short N_r=0;
  unsigned short N_th=0;
  unsigned short N_ph=0;

  // .. load Conf File
  if (options->input!= stdin){

    /**  inserire qui la lettura del file in input. 
     * le chiavi caricate modificano il contenuto di SimParameters
     * poi nella parte verbose si indica cosa è stato caricato dal file 
     **/
    char line[ReadingStringLenght];
    char key[ReadingStringLenght],value[ReadingStringLenght];
    while ((fgets(line, ReadingStringLenght, options->input)) != NULL)
    {
      if (line[0]=='#') continue; // if the line is a comment skip it
      sscanf(line, "%[^:]: %[^\n#]", key, value);                     // for each Key assign the value to correponding Variable
      

      // ------------- file name to be use in output ----------------
      if (strcmp(key,"OutputFilename")==0){ 
        char output_file_name[ReadingStringLenght];
        sprintf(output_file_name,"%s",value);
        if (strlen(output_file_name)>struct_string_lengh-10)
        {
          fprintf (stderr,"ERROR: OutputFilename too long (%d) should be (%d-10)\n",(int)(strlen(output_file_name)),struct_string_lengh);
          return EXIT_FAILURE;
        }
        strncpy(SimParameters.output_file_name,output_file_name,struct_string_lengh); 
      }

      // ------------- Energy binning ----------------
      if (strcmp(key,"Tcentr")==0){ 
        UseDefault_initial_energy=false;
        SimParameters.NT=SplitCSVString(value, &SimParameters.Tcentr);
        if (options->verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### number of input energies from configuration file is %d\n",SimParameters.NT);
          fprintf(stdout,"### energies--> ");
          for (int i_split=0; i_split< SimParameters.NT; i_split++){ 
            fprintf(stdout,"%f ",SimParameters.Tcentr[i_split]);  
          }
          fprintf(stdout,"\n"); 
        }
      }

      // ------------- Number of particle to be simulated ----------------
      if (strcmp(key,"Npart")==0){ 
        SimParameters.Npart= atoi(value);
        if (options->verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### Npart--> %s\n",value);
        }
        if (SimParameters.Npart<=0)
        {
          fprintf (stderr,"ERROR: Npart cannot be 0 or negative \n");
          return EXIT_FAILURE;
        }
      }


      // ------------- Initial position ----------------
      if (strcmp(key,"SourcePos_r")==0){ 
        N_r=SplitCSVString(value, &r);
        if (options->verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### number of input SourcePos_r from configuration file is %d\n",N_r);
          fprintf(stdout,"### SourcePos_r--> ");
          for (int i_split=0; i_split< N_r; i_split++){ fprintf(stdout,"%f ",r[i_split]);  }
          fprintf(stdout,"\n"); 
        }
      }

      if (strcmp(key,"SourcePos_theta")==0){
        N_th=SplitCSVString(value, &th);
        if (options->verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### number of input SourcePos_theta from configuration file is %d\n",N_th);
          fprintf(stdout,"### SourcePos_theta--> ");
          for (int i_split=0; i_split< N_th; i_split++){ 
            fprintf(stdout,"%f ",th[i_split]);  
          }
          fprintf(stdout,"\n"); 
        }
      }

      if (strcmp(key,"SourcePos_phi")==0){ 
        N_ph=SplitCSVString(value, &ph);
        if (options->verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### number of input SourcePos_phi from configuration file is %d\n",N_ph);
          fprintf(stdout,"### SourcePos_phi--> ");
          for (int i_split=0; i_split< N_ph; i_split++){ fprintf(stdout,"%f ",ph[i_split]);  }
          fprintf(stdout,"\n"); 
        }
      }      
      // ------------- particle description ----------------
      if (strcmp(key,"Particle_NucleonRestMass")==0){ 
        SimParameters.IonToBeSimulated.T0= atof(value);
        if (options->verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### Particle_NucleonRestMass--> %s\n",value);
        }
        if (SimParameters.IonToBeSimulated.T0<0)
        {
          fprintf (stderr,"ERROR: Particle_NucleonRestMass cannot be negative \n");
          return EXIT_FAILURE;
        }
      }
      if (strcmp(key,"Particle_MassNumber")==0){ 
        SimParameters.IonToBeSimulated.A = atof(value);
        if (options->verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### Particle_MassNumber--> %s\n",value);
        }
        if (SimParameters.IonToBeSimulated.A<0)
        {
          fprintf (stderr,"ERROR: Particle_MassNumber cannot be negative \n");
          return EXIT_FAILURE;
        }
      }
      if (strcmp(key,"Particle_Charge")==0){ 
        SimParameters.IonToBeSimulated.Z = atof(value);
        if (options->verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### Particle_Charge--> %s\n",value);
        }
      }

      // ------------- Number of region in which divide the heliosphere (Heliosheet excluded) ------
      if (strcmp(key,"Nregions")==0){ 
        Nregions= (unsigned short)atoi(value);
        if (options->verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### Nregions--> %hhu\n",Nregions);
        }
        if (Nregions<=0)
        {
          fprintf (stderr,"ERROR: Nregions cannot be 0 or negative \n");
          return EXIT_FAILURE;
        }
      }

      // ------------- load Zones properties ---------
      if ((strcmp(key,"HeliosphericParameters")==0)&&NumberParameterLoaded<NMaxRegions)
      {
        if (options->verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### HeliosphericParameters--> %s\n",value);
        }
        char delim[] = ",";
        char *token;
        int   i_split=0;
        token = strtok(value, delim);
        while( token != NULL ) 
        {
          switch (i_split)
          {
            case 0:
              IHP[NumberParameterLoaded].k0 = atof(token); 
              if (IHP[NumberParameterLoaded].k0<0) {return PrintError ("k0", value, NumberParameterLoaded);}
              break;
            case 1:
              IHP[NumberParameterLoaded].ssn = atof(token); 
              if (IHP[NumberParameterLoaded].ssn<0) {return PrintError ("SSN", value, NumberParameterLoaded);}
              break;
            case 2: 
              IHP[NumberParameterLoaded].V0 = atof(token); 
              if (IHP[NumberParameterLoaded].V0<0) {return PrintError ("V0", value, NumberParameterLoaded);}
              break;
            case 3: 
              IHP[NumberParameterLoaded].TiltAngle = atof(token); 
              if (IHP[NumberParameterLoaded].TiltAngle<0 || IHP[NumberParameterLoaded].TiltAngle>90) {return PrintError ("TiltAngle", value, NumberParameterLoaded);}
              break;
            case 4: 
              IHP[NumberParameterLoaded].SmoothTilt = atof(token); 
              if (IHP[NumberParameterLoaded].SmoothTilt<0 || IHP[NumberParameterLoaded].SmoothTilt>90) {return PrintError ("SmoothTilt", value, NumberParameterLoaded);}
              break;  
            case 5: 
              IHP[NumberParameterLoaded].BEarth = atof(token); 
              if (IHP[NumberParameterLoaded].BEarth<0. || IHP[NumberParameterLoaded].BEarth>999.) {return PrintError ("BEarth", value, NumberParameterLoaded);}
              break;                           
            case 6: 
              IHP[NumberParameterLoaded].Polarity = atoi(token); 
              if (IHP[NumberParameterLoaded].Polarity!=1. && IHP[NumberParameterLoaded].Polarity!=-1) {return PrintError ("Polarity", value, NumberParameterLoaded);}
              break;  
            case 7: 
              IHP[NumberParameterLoaded].SolarPhase = atoi(token); 
              if (IHP[NumberParameterLoaded].SolarPhase!=1. && IHP[NumberParameterLoaded].SolarPhase!=0) {return PrintError ("Polarity", value, NumberParameterLoaded);}
              break;  
            case 8: 
              IHP[NumberParameterLoaded].NMCR = atof(token); 
              if (IHP[NumberParameterLoaded].NMCR<0. || IHP[NumberParameterLoaded].NMCR>=9999) {return PrintError ("Polarity", value, NumberParameterLoaded);}
              break; 
            case 9:
              IHP[NumberParameterLoaded].Rts_nose = atof(token); 
              if (IHP[NumberParameterLoaded].Rts_nose<=0. || IHP[NumberParameterLoaded].Rts_nose>=999) {return PrintError ("Rts_nose", value, NumberParameterLoaded);}
              break; 
            case 10:
              IHP[NumberParameterLoaded].Rts_tail = atof(token); 
              if (IHP[NumberParameterLoaded].Rts_tail<=0. || IHP[NumberParameterLoaded].Rts_tail>=999) {return PrintError ("Rts_tail", value, NumberParameterLoaded);}
              break;   
            case 11:
              IHP[NumberParameterLoaded].Rhp_nose = atof(token); 
              if (IHP[NumberParameterLoaded].Rhp_nose<=0. || IHP[NumberParameterLoaded].Rhp_nose>=999) {return PrintError ("Rhp_nose", value, NumberParameterLoaded);}
              break; 
            case 12:
              IHP[NumberParameterLoaded].Rhp_tail = atof(token); 
              if (IHP[NumberParameterLoaded].Rhp_tail<=0. || IHP[NumberParameterLoaded].Rhp_tail>=999) {return PrintError ("Rhp_tail", value, NumberParameterLoaded);}
              break; 
          }
          token = strtok(NULL, delim);
          i_split++;
        }
        NumberParameterLoaded++;
      }

// ------------- load Zones properties ---------
      if ((strcmp(key,"HeliosheatParameters")==0)&&NumberHeliosheatParLoaded<NMaxRegions)
      {
        if (options->verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### HeliosheatParameters--> %s\n",value);
        }
        char delim[] = ",";
        char *token;
        int   i_split=0;
        token = strtok(value, delim);
        while( token != NULL ) 
        {
          switch (i_split)
          {
            case 0:
              IHS[NumberHeliosheatParLoaded].k0 = atof(token); 
              if (IHS[NumberHeliosheatParLoaded].k0<=0) {return PrintError ("k0", value, NumberHeliosheatParLoaded);}
              break;
            case 1: 
              IHS[NumberHeliosheatParLoaded].V0 = atof(token); 
              if (IHS[NumberHeliosheatParLoaded].V0<=0) {return PrintError ("V0", value, NumberHeliosheatParLoaded);}
              break;
          }
          token = strtok(NULL, delim);
          i_split++;
        }
        NumberHeliosheatParLoaded++;
      }
      // ------------- Output controls ---------------
      if (strcmp(key,"RelativeBinAmplitude")==0){ 
        SimParameters.RelativeBinAmplitude= atof(value);
        if (options->verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### RelativeBinAmplitude--> %s\n",value);
        }
        if (SimParameters.RelativeBinAmplitude>0.01)
        {
          fprintf (stderr,"ERROR: RelativeBinAmplitude cannot be greater than 1%% (0.01) if you whish to have decent results. \n");
          return EXIT_FAILURE;
        }
      }
    }// ------------- END parsing ----------------



    if (options->verbose) { 
      fprintf(stderr,LOAD_CONF_FILE_SiFile);
    }
    fclose(options->input);
  }else{
    if (options->verbose) { 
      fprintf(stderr,LOAD_CONF_FILE_NoFile);
    }
  }


  //.. compose initial Position array
  // check that Npositions are the same for all coordinates
  if ( (N_r!=N_th) || (N_r!=N_ph) )
  {
    fprintf(stderr,"ERROR:: the number of initial coodinates is different Nradius=%hhu Ntheta=%hhu Nphi=%hhu\n",N_r,N_th,N_ph); 
    return EXIT_FAILURE; // in this case the initial position is ambiguous
  }
  // initialize the initial position array --> NOTE: SimParameters.NInitialPositions correspond to number of periods (Carrington rotation) to be simulated
  SimParameters.NInitialPositions = N_r;
  SimParameters.InitialPosition   = (vect3D_t*)malloc( N_r * sizeof(vect3D_t) );
  for (int iPos=0; iPos<N_r; iPos++)
  {
    // validity check
    if (r[iPos]<=SimParameters.HeliosphereToBeSimulated.Rmirror)
    {
      fprintf (stderr,"ERROR: check %dth value of SourcePos_r because it cannot be smaller than %.1f \n",iPos,SimParameters.HeliosphereToBeSimulated.Rmirror);
      return EXIT_FAILURE;      
    }
    if (fabs(th[iPos])>Pi)
    {
      fprintf (stderr,"ERROR: check %dth value of SourcePos_theta cannot be greater than +%.1f \n",iPos,Pi);
      return EXIT_FAILURE;
    }
    if ((ph[iPos]<0) || (ph[iPos]>Pi))
    {
      fprintf (stderr,"ERROR: check %dth value of SourcePos_phi cannot be ouside the interval [0,%.1f] \n",iPos,Pi);
      return EXIT_FAILURE;
    }
    // insert the value
    SimParameters.InitialPosition[iPos].r  =r[iPos];
    SimParameters.InitialPosition[iPos].th =th[iPos];
    SimParameters.InitialPosition[iPos].phi=ph[iPos];
  }

  //check if the number of loaded region is sufficiend
  if (NumberParameterLoaded<N_r+Nregions-1)
  {
    fprintf (stderr,"ERROR: Too few heliospheric parameter to cover the desidered period and regions \n");
    fprintf (stderr,"ERROR: Loaded Parameter regions = %d ; Source positions = %d ; No. of heliospheric region = %hhu\n",NumberParameterLoaded,N_r,Nregions);
    return EXIT_FAILURE;
  }
  if (NumberHeliosheatParLoaded<N_r)
  {
    fprintf (stderr,"ERROR: Too few Heliosheat parameters to cover the desidered period and regions \n");
    fprintf (stderr,"ERROR: Loaded Parameter regions = %d ; Source positions = %d ; \n",NumberHeliosheatParLoaded,N_r);
    return EXIT_FAILURE;
  }
  // .. Set if is High Activity Period and Radial boundaried

  for (int iPos=0; iPos<N_r; iPos++)
  {  
    // .. Set if is High Activity Period
    float AverTilt = 0;
    for (int izone =0 ; izone<Nregions; izone++)
      { 
        AverTilt+=IHP[izone+iPos].TiltAngle;
      }
    SimParameters.HeliosphereToBeSimulated.IsHighActivityPeriod[iPos]= (AverTilt/float(Nregions)>=TiltL_MaxActivity_threshold)?true:false ;
    // .. radial boundaries
    SimParameters.HeliosphereToBeSimulated.RadBoundary_real[iPos].Rts_nose=IHP[iPos].Rts_nose;
    SimParameters.HeliosphereToBeSimulated.RadBoundary_real[iPos].Rts_tail=IHP[iPos].Rts_tail;
    SimParameters.HeliosphereToBeSimulated.RadBoundary_real[iPos].Rhp_nose=IHP[iPos].Rhp_nose;
    SimParameters.HeliosphereToBeSimulated.RadBoundary_real[iPos].Rhp_tail=IHP[iPos].Rhp_tail;
  }
  // .. Fill Heliosphere
  SimParameters.HeliosphereToBeSimulated.Nregions = Nregions; 
  for (int izone =0 ; izone<NumberParameterLoaded; izone++){
    SimParameters.prop_medium[izone].V0=IHP[izone].V0/aukm;
    if (IHP[izone].k0>0) 
    {
      SimParameters.prop_medium[izone].k0_paral[0] = IHP[izone].k0*K0CorrFactor(int(IHP[izone].Polarity), 
                                                                                int(SimParameters.IonToBeSimulated.Z), 
                                                                                int(IHP[izone].SolarPhase), 
                                                                                IHP[izone].SmoothTilt);
      SimParameters.prop_medium[izone].k0_paral[1] = IHP[izone].k0*K0CorrFactor(int(IHP[izone].Polarity), 
                                                                                int(SimParameters.IonToBeSimulated.Z), 
                                                                                int(IHP[izone].SolarPhase), 
                                                                                IHP[izone].SmoothTilt);
      SimParameters.prop_medium[izone].k0_perp[0]  = IHP[izone].k0;
      SimParameters.prop_medium[izone].k0_perp[1]  = IHP[izone].k0;
      SimParameters.prop_medium[izone].GaussVar[0] = 0;
      SimParameters.prop_medium[izone].GaussVar[1] = 0;
    }else{
      float3 K0 = EvalK0(true, // isHighActivity
                        IHP[izone].Polarity, 
                        SimParameters.IonToBeSimulated.Z, 
                        IHP[izone].SolarPhase, 
                        IHP[izone].SmoothTilt, 
                        IHP[izone].NMCR,
                        IHP[izone].ssn, 
                        options->verbose);
      SimParameters.prop_medium[izone].k0_paral[0] = K0.x;
      SimParameters.prop_medium[izone].k0_perp[0]  = K0.y;
      SimParameters.prop_medium[izone].GaussVar[0] = K0.z;
      K0 = EvalK0(false, // isHighActivity 
                        IHP[izone].Polarity, 
                        SimParameters.IonToBeSimulated.Z, 
                        IHP[izone].SolarPhase, 
                        IHP[izone].SmoothTilt, 
                        IHP[izone].NMCR,
                        IHP[izone].ssn, 
                        options->verbose);
      SimParameters.prop_medium[izone].k0_paral[1] = K0.x;
      SimParameters.prop_medium[izone].k0_perp[1]  = K0.y;
      SimParameters.prop_medium[izone].GaussVar[1] = K0.z;
    }
    SimParameters.prop_medium[izone].g_low = g_low(IHP[izone].SolarPhase, IHP[izone].Polarity, IHP[izone].SmoothTilt);
    SimParameters.prop_medium[izone].rconst= rconst(IHP[izone].SolarPhase, IHP[izone].Polarity, IHP[izone].SmoothTilt);
    SimParameters.prop_medium[izone].TiltAngle = IHP[izone].TiltAngle*Pi/180.; // conversion to radian
//bugfixed------------> 
    SimParameters.prop_medium[izone].Asun  = float(IHP[izone].Polarity)*(aum*aum)*IHP[izone].BEarth*1e-9/sqrt( 1.+ ((Omega*(1-rhelio))/(IHP[izone].V0/aukm))*((Omega*(1-rhelio))/(IHP[izone].V0/aukm)) ) ;

//HelMod-Like
  //SimParameters.prop_medium[izone].Asun  = float(IHP[izone].Polarity)*(aum*aum)*IHP[izone].BEarth*1e-9/sqrt( 1.+ ((Omega*(1-rhelio))/(IHP[0].V0/aukm))*((Omega*(1-rhelio))/(IHP[0].V0/aukm)) ) ;
    //fprintf(stderr,"Asun --> %e %e %e\n",1.,IHP[izone].BEarth, float(IHP[izone].Polarity)*(aum*aum)*IHP[izone].BEarth*1e-9/sqrt( 1.+ ((Omega*(1-rhelio))/(IHP[izone].V0/aukm))*((Omega*(1-rhelio))/(IHP[izone].V0/aukm)) ));
    SimParameters.prop_medium[izone].P0d   = EvalP0DriftSuppressionFactor(0,IHP[izone].SolarPhase,IHP[izone].TiltAngle,0);
    SimParameters.prop_medium[izone].P0dNS = EvalP0DriftSuppressionFactor(1,IHP[izone].SolarPhase,IHP[izone].TiltAngle,IHP[izone].ssn);
    SimParameters.prop_medium[izone].plateau = EvalHighRigidityDriftSuppression_plateau(IHP[izone].SolarPhase, IHP[izone].TiltAngle);
  }
  // .. Fill Heliosheat
  for (int izone =0 ; izone<NumberHeliosheatParLoaded; izone++){
    SimParameters.prop_Heliosheat[izone].k0=IHS[izone].k0;
    SimParameters.prop_Heliosheat[izone].V0=IHS[izone].V0/aukm;
  }

  // .. init variable with default values 
  if (UseDefault_initial_energy){
    /* init the energy binning*/
    SimParameters.NT=10;
    SimParameters.Tcentr = (float*)malloc( SimParameters.NT * sizeof(float) );
    float Tmin = .1;   
    float Tmax = 100.;
    // log binning
    float dlT_Log=log10(Tmax/Tmin)/((float)SimParameters.NT);              /* step of log(T)*/
    float X=log10(Tmax);                                 /*esponente per energia*/
    for (int j=0; j<SimParameters.NT; j++)
    {
      float tem=X-(j+1)*dlT_Log;                          /*exponent */
      // Ts=pow(10.0,tem);                                 /* bin border */
      SimParameters.Tcentr[j]=sqrt(pow(10.0,tem)*pow(10.0,(tem+dlT_Log)));       /* geom.centre of bin */
      if (options->verbose>=VERBOSE_hig) {fprintf(stdout,"### BIN::\t%d\t%.2f\n",j,SimParameters.Tcentr[j]);}
    }
  }



  // .. init other variables
  SimParameters.Results                 = (MonteCarloResult_t*)malloc( SimParameters.NT * sizeof(MonteCarloResult_t) );
 

  // .. recap simulation parameters  - print SimParameters_t content
  if (options->verbose>=VERBOSE_med) { 
      fprintf(stderr,"----- Recap of Simulation parameters ----\n");
      fprintf(stderr,"NucleonRestMass         : %.3f Gev/n \n",SimParameters.IonToBeSimulated.T0);
      fprintf(stderr,"MassNumber              : %.1f \n",SimParameters.IonToBeSimulated.A);
      fprintf(stderr,"Charge                  : %.1f \n",SimParameters.IonToBeSimulated.Z);
      fprintf(stderr,"Number of sources       : %hhu \n",SimParameters.NInitialPositions);
      for (int ipos=0 ; ipos<SimParameters.NInitialPositions; ipos++)
      {
        fprintf(stderr,"position              :%d \n",ipos);
        fprintf(stderr,"  Init Pos (real) - r     : %.2f \n",SimParameters.InitialPosition[ipos].r);
        fprintf(stderr,"  Init Pos (real) - theta : %.2f \n",SimParameters.InitialPosition[ipos].th);
        fprintf(stderr,"  Init Pos (real) - phi   : %.2f \n",SimParameters.InitialPosition[ipos].phi);
      }
      fprintf(stderr,"output_file_name        : %s \n",SimParameters.output_file_name);
      fprintf(stderr,"number of input energies: %d \n",SimParameters.NT);
      fprintf(stderr,"input energies          : ");
      for (int itemp=0; itemp<SimParameters.NT; itemp++) { fprintf(stderr,"%.2f ",SimParameters.Tcentr[itemp]); }
      fprintf(stderr,"\n"); 
      fprintf(stderr,"Events to be generated  : %lu \n",SimParameters.Npart);
      //fprintf(stderr,"Warp per Block          : %d \n",WarpPerBlock);

      fprintf(stderr,"\n"); 
      fprintf(stderr,"for each simulated periods:\n");
      for (int ipos=0 ; ipos<SimParameters.NInitialPositions; ipos++)
      {
        fprintf(stderr,"position              :%d \n",ipos);
        fprintf(stderr,"  IsHighActivityPeriod    : %s \n",SimParameters.HeliosphereToBeSimulated.IsHighActivityPeriod[ipos] ? "true" : "false");
        fprintf(stderr,"  Rts nose direction      : %.2f AU\n",SimParameters.HeliosphereToBeSimulated.RadBoundary_real[ipos].Rts_nose);
        fprintf(stderr,"  Rts tail direction      : %.2f AU\n",SimParameters.HeliosphereToBeSimulated.RadBoundary_real[ipos].Rts_tail);
        fprintf(stderr,"  Rhp nose direction      : %.2f AU\n",SimParameters.HeliosphereToBeSimulated.RadBoundary_real[ipos].Rhp_nose);
        fprintf(stderr,"  Rhp tail direction      : %.2f AU\n",SimParameters.HeliosphereToBeSimulated.RadBoundary_real[ipos].Rhp_tail);
      }
      fprintf(stderr,"Heliopshere Parameters ( %d regions ): \n",SimParameters.HeliosphereToBeSimulated.Nregions);
      
      for (int iregion=0 ; iregion<SimParameters.HeliosphereToBeSimulated.Nregions+SimParameters.NInitialPositions-1 ; iregion++)
      {
        fprintf(stderr,"- Region %d \n",iregion);
        fprintf(stderr,"-- V0         %e AU/s\n",SimParameters.prop_medium[iregion].V0);
        fprintf(stderr,"-- k0_paral   [%e,%e] \n",SimParameters.prop_medium[iregion].k0_paral[0],SimParameters.prop_medium[iregion].k0_paral[1]);
        fprintf(stderr,"-- k0_perp    [%e,%e] \n",SimParameters.prop_medium[iregion].k0_perp[0],SimParameters.prop_medium[iregion].k0_perp[1]);
        fprintf(stderr,"-- GaussVar   [%.4f,%.4f] \n",SimParameters.prop_medium[iregion].GaussVar[0],SimParameters.prop_medium[iregion].GaussVar[1]);
        fprintf(stderr,"-- g_low      %.4f \n",SimParameters.prop_medium[iregion].g_low);
        fprintf(stderr,"-- rconst     %.3f \n",SimParameters.prop_medium[iregion].rconst);
        fprintf(stderr,"-- tilt angle %.3f rad\n",SimParameters.prop_medium[iregion].TiltAngle);
        fprintf(stderr,"-- Asun       %e \n",SimParameters.prop_medium[iregion].Asun);
        fprintf(stderr,"-- P0d        %e GV \n",SimParameters.prop_medium[iregion].P0d);
        fprintf(stderr,"-- P0dNS      %e GV \n",SimParameters.prop_medium[iregion].P0dNS);

        
        // XXXXXXX

      }
      fprintf(stderr,"Heliosheat parameters ( %d periods ): \n",SimParameters.NInitialPositions);
      for (int ipos=0 ; ipos<SimParameters.NInitialPositions; ipos++)
      {
        fprintf(stderr,"-period              :%d \n",ipos);
        fprintf(stderr,"-- V0 %e AU/s\n",SimParameters.prop_Heliosheat[ipos].V0);
        fprintf(stderr,"-- k0 %e \n",SimParameters.prop_Heliosheat[ipos].k0);
      }
      fprintf(stderr,"----------------------------------------\n"); 
    /** XXXX inserire qui il recap dei parametri di simulazione  
     * 
     **/
  }
  // .. final checks
  if (SimParameters.HeliosphereToBeSimulated.Nregions<1) {
    fprintf(stderr,"ERROR::not enough regions loaded, must be at least 2 (1 below TS and 1 for Heliosheat)\n"); 
    return EXIT_FAILURE; // in this case no regions were loaded
  }

  return EXIT_SUCCESS;
}


////////////////////////////////////////////////////////////////
//..... GPU useful and safe function ...........................
//  
////////////////////////////////////////////////////////////////

// ... manage error in GPU 
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

// ... ceil safe division for integer. Rounds x=a/b upward, returning the smallest integral value that is not less than x.
int ceil_int(int a, int b){
  // https://www.reddit.com/r/C_Programming/comments/gqpuef/comment/fru7tmu/?utm_source=share&utm_medium=web2x&context=3
  return ((a+(b-1))/b);
}

// ... floor safe division for integer. Rounds x=a/b downward, returning the biggest integral value that is less than x.
int floor_int(int a, int b){
  return int(floor(a/b));
}

// find minimums d in array a (note that d is an array of lenght nblock)
__global__ void kernel_max(particle_t *a, float *d, unsigned long Npart)
{
  __shared__ float sdata[tpb]; //"static" shared memory

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<Npart){
    sdata[tid] = a[i].part.Ek;
  }
  // if (blockIdx.x ==15 && threadIdx.x==199)
  // {
  //   printf("lll %u %.2f %.2f \n",i,a[i].part.Ek,sdata[tid]);
  // }
  __syncthreads();
  for(unsigned int s=tpb/2 ; s >= 1 ; s=s/2)
  {
    if(tid < s && i<Npart)
    {
      if(sdata[tid] < sdata[tid + s])
      {
        sdata[tid] = sdata[tid + s];
      }
    }
    __syncthreads();
  }
  if(tid == 0 ) 
  {
    d[blockIdx.x] = sdata[0];
  }
}

////////////////////////////////////////////////////////////////
//..... Random generator .......................................
//  
////////////////////////////////////////////////////////////////
__global__ void init_rdmgenerator(curandStatePhilox4_32_10_t *state, unsigned long long seed )
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(seed, id, 0, &state[id]);
}

////////////////////////////////////////////////////////////////
//..... histogram handling .....................................
//  
////////////////////////////////////////////////////////////////
__global__ void histogram_atomic(const particle_t *in, 
                                 const float LogBin0_lowEdge, 
                                 const float DeltaLogT , 
                                 const int Nbin, 
                                 const unsigned long Npart,  
                                 float *out,
                                 int *Nfailed )
{
  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  
  for (int it=threadIdx.x ; it<Nbin; it+=blockDim.x) 
  {
    out[blockIdx.x*Nbin+it]=0;
  }
  __syncthreads();
  
  if ( (id<Npart)  )
  { 
    if (log10(in[id].part.Ek)>LogBin0_lowEdge){
      int DestBin = floor( (log10(in[id].part.Ek)-LogBin0_lowEdge)/DeltaLogT ); // evalaute the bin where put event
      atomicAdd(&out[blockIdx.x*Nbin+DestBin], exp(in[id].alphapath));
    }else{
      atomicAdd(Nfailed, 1); // nota per futuro. le particelle uccise hanno valori diversi negativi in base all'evento che li ha uccisi, 
                             //                  quindi se Nfailed diventasse una struct con il tipo di errore, si potrebbe fare una 
                             //                  statistica dettagliata dell'errore.
    }
  }
}

__global__ void histogram_accum(const float *in,  const int Nbins, const int NFraz, float *out)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id>=Nbins) { return;} //out of range
  float total=0.;
  for (int ithb=0; ithb<NFraz; ithb++){
    total+=in[id+ithb*Nbins];
  }
  out[id]=total;
}


////////////////////////////////////////////////////////////////
//..... Heliospheric propagation ...............................
//  
////////////////////////////////////////////////////////////////
// usage goal  = 64 register * 32-bit = 2048 bit

__global__ void HeliosphericPropagation(curandStatePhilox4_32_10_t *state, PropagationParameters_t Param,particle_t *FinalPropState, int* PeriodIndexArray ) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  //printf("%d\n",id);
  if (id<Param.NpartPerKernelExecution)                         // ensure the particle have a correct id
  {
    //printf("%d\n",id);
    curandStatePhilox4_32_10_t localState = state[id];          /* Copy state to local memory for efficiency */
    particle_t Quasi_Particle_Object = FinalPropState[id];
    int PeriodIndex                  = PeriodIndexArray[id];
    // REMIND PeriodIndex is the period we are simulating (to be added to Heliosphere_Region_Number when loading parameters )
    signed short Heliosphere_Region_Number = RadialZone(PeriodIndex,
                                                       Quasi_Particle_Object.part);

    while (Heliosphere_Region_Number>=0)
    {
      // Evaluate Random num
      float4 RndNum = curand_normal4(&localState); // x,y,z used for SDE, w used for K0 random oscillation
// // to be del XXXXXXXXXXXXXXX
// RndNum.w=0; ///<------ for test XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

      // Evaluate conv-diff coeff
      DiffusionTensor_t K = DiffusionTensor_symmetric( PeriodIndex,
                                                       Heliosphere_Region_Number, 
                                                       Quasi_Particle_Object.part,
                                                       Quasi_Particle_Object.pt,
                                                       RndNum.w );
      int res=0;
                                                                                                    // if (id==0) {
                                                                                                    //   printf("+%.2f Krr=%e\tKtt=%e\tKrt=%e\n",Quasi_Particle_Object.part.r,K.K.rr,K.K.tt,K.K.tr);
                                                                                                     
                                                                                                    // }
      Tensor3D_t Ddif=SquareRoot_DiffusionTerm(Heliosphere_Region_Number,
                                               K.K,
                                               Quasi_Particle_Object.part,res);   
                                                                                                    // // ------------- DEBUG LINES!!!
                                                                                                    // if (id==0) {
                                                                                                    //   printf("+%.2f Krr=%e\tKtt=%e\tKrt=%e\tKpr=%e\tKpp=%e\n",Quasi_Particle_Object.part.r,K.K.rr,K.K.tt,K.K.tr,K.K.pr,K.K.pp);
                                                                                                    //   printf("+%.2f DKrr_dr=%e\tDKrp_dr=%e\n",Quasi_Particle_Object.part.r,K.DKrr_dr,K.DKrp_dr);
                                                                                                    //   printf("+     - res=%d - Ddif_rr=%e\tDdif_tr=%e\tDdif_pr=%e\n",res,Ddif.rr,Ddif.tr,Ddif.pr);
                                                                                                    //   printf("+     -          Ddif_tt=%e\tDdif_pt=%e\tDdif_pp=%e\n",res,Ddif.tt,Ddif.pt,Ddif.pp);
                                                                                                    // }
                                                                                                    // // ----------------------------
      if (res>0)
      {
        // SDE diffusion matrix is not positive definite; in this case propagation should be stopped and a new event generated
        // placing the energy below zero ensure that this event is ignored in the after-part of the analysis
        Quasi_Particle_Object.part.Ek=-1;   
        break; //exit the while cycle 
      }

      // Evaluate conv-drift velocity 
      vect3D_t AdvTerm = AdvectiveTerm(PeriodIndex,
                                       Heliosphere_Region_Number, 
                                       K,
                                       Quasi_Particle_Object.part,
                                       Quasi_Particle_Object.pt);


      // evaluate time step
      float dt = MaxValueTimeStep;
      // time step is modified to ensure the diffusion approximation (i.e. diffusion step>>advective step)
      if (dt>MinValueTimeStep * (Ddif.rr*Ddif.rr)/(AdvTerm.r*AdvTerm.r))                     dt=max(MinValueTimeStep, MinValueTimeStep * (Ddif.rr*Ddif.rr)                  /(AdvTerm.r*AdvTerm.r));
      if (dt>MinValueTimeStep * (Ddif.tr+Ddif.tt)*(Ddif.tr+Ddif.tt)/(AdvTerm.th*AdvTerm.th)) dt=max(MinValueTimeStep, MinValueTimeStep * (Ddif.tr+Ddif.tt)*(Ddif.tr+Ddif.tt)/(AdvTerm.th*AdvTerm.th));
      //if (dt>MinValueTimeStep * (Ddif.pr+Ddif.pt+Ddif.pp)*(Ddif.pr+Ddif.pt+Ddif.pp)/(AdvTerm.phi*AdvTerm.phi)) dt=max(MinValueTimeStep, MinValueTimeStep * (Ddif.pr+Ddif.pt+Ddif.pp)*(Ddif.pr+Ddif.pt+Ddif.pp)/(AdvTerm.phi*AdvTerm.phi));


      // evalaute particle step
      particle_t prevStep=Quasi_Particle_Object;


// Quasi_Particle_Object.part.Ek  +=0; 
// Quasi_Particle_Object.alphapath+=LossTerm(Heliosphere_Region_Number, Quasi_Particle_Object )*dt;; 
// Quasi_Particle_Object.part.r   +=10; 
// Quasi_Particle_Object.part.th  +=0; 
// Quasi_Particle_Object.part.phi +=0; 
// Quasi_Particle_Object.prop_time+=dt;
      Quasi_Particle_Object.part.Ek  += EnergyLoss(PeriodIndex,
                                                         Heliosphere_Region_Number, 
                                                         Quasi_Particle_Object)*dt;
      Quasi_Particle_Object.alphapath+= LossTerm(PeriodIndex,
                                                       Heliosphere_Region_Number, 
                                                       Quasi_Particle_Object )*dt;
      Quasi_Particle_Object.part.r   += AdvTerm.r*dt   + (RndNum.x*Ddif.rr)*sqrt(dt);
      Quasi_Particle_Object.part.th  += AdvTerm.th*dt  + (RndNum.x*Ddif.tr+RndNum.y*Ddif.tt)*sqrt(dt);
      Quasi_Particle_Object.part.phi += AdvTerm.phi*dt + (RndNum.x*Ddif.pr+RndNum.y*Ddif.pt+RndNum.z*Ddif.pp)*sqrt(dt);
      Quasi_Particle_Object.prop_time+= dt;


      // post step checks

      // --- Mirroring -- if particle reach inner bounduary redo last step
      if (Quasi_Particle_Object.part.r<Heliosphere.Rmirror)
      {
        Quasi_Particle_Object=prevStep;
        continue;
      }
      // --- reflecting latitudinal bounduary
      if (Quasi_Particle_Object.part.th>thetaSouthlimit) 
        {Quasi_Particle_Object.part.th = 2*thetaSouthlimit-Quasi_Particle_Object.part.th;}
      else if (Quasi_Particle_Object.part.th<(thetaNorthlimit))    
        {Quasi_Particle_Object.part.th = 2*thetaNorthlimit-Quasi_Particle_Object.part.th;}

      // --- mantain the azimut in module 2PI
      while (Quasi_Particle_Object.part.phi>2.*Pi) { Quasi_Particle_Object.part.phi=Quasi_Particle_Object.part.phi-2.*Pi;  }
      while (Quasi_Particle_Object.part.phi<0)     { Quasi_Particle_Object.part.phi=Quasi_Particle_Object.part.phi+2.*Pi;  }

      
      // find Heliospheric region, if Ek<=0 some error occours and particle is killed
      if (Quasi_Particle_Object.part.Ek>0){
        Heliosphere_Region_Number = RadialZone(PeriodIndex,
                                               Quasi_Particle_Object.part);
      }else{
        Heliosphere_Region_Number = -1;
      }
    }

    // Save result
    FinalPropState[id]=Quasi_Particle_Object;
    // salva lo stato finale <---------------- serve specialmente se si pensa di riutilizarlo, in questo modo si evita di riutilizzare le stesse sequenze
    state[id] = localState; /* Copy state back to global memory */
  }
}


