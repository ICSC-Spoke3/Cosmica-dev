#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include <stdarg.h>
#include <time.h>

// .. CUDA specific
#include <curand.h>         // CUDA random number host library
#include <curand_kernel.h>  // CUDA random number device library
#include <cuda_runtime.h>   // Device code management by providing implicit initialization, context management, and module management

#define OPTSTR "vi:h"

////////////////////////////////////////////////////////////////
//..... Propagation variables and struct .......................
////////////////////////////////////////////////////////////////

/*Constant*/
// const float V0 = 2.66667e-6;
// const float K0 = 0.000222;
// const float T0 = 0.938272;
// const float max_dt = 50.;
// const float RHP = 100.;

// const int Npart = 5024;
// const int NinitE = 11;
// const float initE[11] = {4.924e-01,6.207e-01,7.637e-01,9.252e-01,1.105e+00,2.103e+01,3.947e+01,7.137e+01,1.291e+02,2.741e+02,1.464e+03};
// const int Npos = 35;
// float initr[Npos] = {0};
// const float RelativeBinAmplitude = 0.00855;

int seed;

// Data container for output result of a single energy simulation
typedef struct MonteCarloResult_t {
  unsigned long Nregistered;
  int           Nbins;
  float         LogBin0_lowEdge;  // lower boundary of first bin
  float         DeltaLogT;        // Bin amplitude in log scale
  float         *BoundaryDistribution;
} MonteCarloResult_t;

// Place here all simulation variables
typedef struct SimParameters_t {
  char          output_file_name[2000]="SimTest";
  int           Npart=5024;                      // number of event to be simulated
  int           NT;                              // number of bins of energies to be simulated
  int           NInitialPositions=0;             // number of initial positions -> this number represent also the number of Carrington rotation that                 
  float         *Tcentr;                         // array of energies to be simulated
  float         *InitialPosition;                // initial position
  float         T0;                              // Particle rest mass
  float         RelativeBinAmplitude = 0.00855 ; // relative (respect 1.) amplitude of Energy bin used as X axis in BoundaryDistribution  --> delta T = T*RelativeBinAmplitude
  float         V0 = 2.66667e-6;                 // Solar wind speed constant
  float         K0 = 0.000222;                   // Diffusion parameter trivial
  float         max_dt = 50.;                    // Time step of propagation
  float         RHP = 100.;                      // Radius of the heliosphere
} SimParameters_t;

////////////////////////////////////////////////////////////////
//..... Input file functions ...................................
////////////////////////////////////////////////////////////////

// Reading and spliting line
unsigned char SplitCSVString(const char *InputString, float **Outputarray)
{
  unsigned char Nelements=0;
  char delim[] = ",";
  char *token;
  char cp_value[2000];
  strncpy(cp_value,InputString,2000);  // strtok modify the original string, since wehave to use it twice, we need a copy of "value"
        
  // ......  read first time the string and count the number of energies
  int i_split=0;
  token = strtok(cp_value, delim);
  while( token != NULL ) 
  {
    token = strtok(NULL, delim);
    i_split++;
  }
  Nelements=i_split;
  // ...... Read again and save value
  *Outputarray = (float*)malloc( Nelements * sizeof(float) );
  i_split=0;
  strncpy(cp_value,InputString,2000);
  token = strtok(cp_value, delim);
  while( token != NULL ) 
  {
    (*Outputarray)[i_split]= atof(token);
    token = strtok(NULL, delim);
    i_split++;
  }
  free(token);
  
  return Nelements;
}

// Load input reduced for test with protons and positrons
void Load_Configuration_File(int argc, char* argv[], struct SimParameters_t &SimParameters) {
  FILE *input=stdin;
  opterr = 0;

  // .. load arguments
  int opt; 
  while ((opt = getopt(argc, argv, OPTSTR)) != EOF)
    switch(opt) {
      case 'i':
        if (!(input = fopen(optarg, "r")) ){
          perror(optarg);
          exit(EXIT_FAILURE);
          /* NOTREACHED */
        }
        break;
    }

  // .. load Conf File
  if (input!= stdin){
    char line[2000];
    char key[2000],value[2000];

    while ((fgets(line, 2000, input)) != NULL) {
      if (line[0]=='#') continue; // if the line is a comment skip it
      sscanf(line, "%[^:]: %[^\n#]", key, value); // for each Key assign the value to correponding Variable

      // ------------- file name to be use in output ----------------
      if (strcmp(key,"OutputFilename")==0){ 
        char output_file_name[2000];
        sprintf(output_file_name,"%s",value);
        strncpy(SimParameters.output_file_name,output_file_name,70); 
      }

      // ------------- Energy binning ----------------
      if (strcmp(key,"Tcentr")==0){ 
        SimParameters.NT=SplitCSVString(value, &SimParameters.Tcentr);
      }

      // ------------- Number of particle to be simulated ----------------
      if (strcmp(key,"Npart")==0){ 
        SimParameters.Npart= atoi(value);
      }

      // ------------- Initial position ----------------
      if (strcmp(key,"SourcePos_r")==0){ 
        SimParameters.NInitialPositions=SplitCSVString(value, &SimParameters.InitialPosition);
      }

      // ------------- particle description ----------------
      if (strcmp(key,"Particle_NucleonRestMass")==0){ 
        SimParameters.T0= atof(value);
      }
    }
  }
}

////////////////////////////////////////////////////////////////
//..... General functions ......................................
////////////////////////////////////////////////////////////////

// Find maximum in array
float findMax(float* arr, int size) {
  float max = arr[0]; // Assume first element is the maximum
  for (int i = 1; i < size; i++) {
      if (arr[i] > max) {
          max = arr[i]; // Update max if current element is larger
      }
  }
  return max;
}

// Print GPU device array
__device__ __host__ void printGPU(float* arr, int size){
  for (int i=0; i<size; i++) {
    printf("arr[%d] = %f\n", i, arr[i]);
  }
}

// Save propagation initial and final particle state
void SaveTxt_part(const char* filename, int Npart, float* r_in, float* T_in, float* r_out, float* T_out, float* alphapath) {

  FILE* file = fopen(filename, "ab");
  if (file == NULL) {
      printf("Error opening the file %s\n", filename);
      exit(EXIT_FAILURE);
  }

  else {
    fprintf(file, "r_in\t T_in\t r_out\t T_out\t alphapath\n");
    
    for (int i=0; i<Npart; i++) {
        fprintf(file, "%f\t %f\t %f\t %f\t %f\n", r_in[i], T_in[i], r_out[i], T_out[i], alphapath[i]);
    }
  }

  fclose(file);
}

////////////////////////////////////////////////////////////////
//..... Random generator .......................................
////////////////////////////////////////////////////////////////

__global__ void init_rdmgenerator(curandStatePhilox4_32_10_t *state, unsigned long seed) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, id, 0, &state[id]);
}

__global__ void rdmgenerator(curandStatePhilox4_32_10_t* state, float* outRandNum, int id) {
  *outRandNum = curand_normal(&state[id]);
}

__global__ void printPhiloxState(curandStatePhilox4_32_10_t *state, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        printf("Key.x: %u\n", state[tid].key.x);
        printf("Key.y: %u\n", state[tid].key.y);
    }
}

////////////////////////////////////////////////////////////////
//..... Propagation kernel .....................................
////////////////////////////////////////////////////////////////

__global__ void propagation(curandStatePhilox4_32_10_t* state, float* outr, float* outE, float* alpha, SimParameters_t SimParameters) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  int Npart = SimParameters.Npart;
  int RHP = SimParameters.RHP;
  int max_dt = SimParameters.max_dt;

  if (id<Npart) {
    float r = outr[id];
    float T = outE[id];
    float alphapath = alpha[id];
    float T0 = SimParameters.T0;
    float V0 = SimParameters.V0;
    float K0 = SimParameters.K0;

    while(r<RHP) {
      float dt = max_dt;
      float RandNum = curand_normal(&state[id]);
      
      float R = sqrt(T*(T+2.*T0));
      float beta = sqrt(T*(T + T0 + T0))/(T + T0);
      
      float Ddif = K0*beta*R;
      float AdvTerm = 2.*Ddif/r - V0;
      float en_loss = 2.*V0*(T+2.*T0)/(T+T0)*T/(3.*r);
      float loss_term = 2.*V0/r *( 1./3.*( T*T+2.*T*T0+ 2.*T0*T0 )/( (T+T0)*(T+T0) ) -1.);

      float prev_r = r;

      r += AdvTerm*dt + RandNum*sqrt(2*Ddif*dt);

      if (r<0.3) {
        r = prev_r;
        continue;
      }

      T += en_loss*dt;
      alphapath += loss_term*dt;
    }

    outr[id] = r;
    outE[id] = T;
    alpha[id] = alphapath;
  }
}

////////////////////////////////////////////////////////////////
//..... MAIN ...................................................
////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {

  //..... Initialization .....//

  // Input file reading and simulation initialization
  struct SimParameters_t SimParameters;
  Load_Configuration_File(argc, argv, SimParameters);

  /* Seed random number generator */
  seed = time(NULL);
  srand(seed);

  // Initial position initialization to earth position (r = 1 AU)
  for (int i= 0; i<SimParameters.NInitialPositions; i++) {
    SimParameters.InitialPosition[i] = 1.;
  }

  // PArticle history save file
  char final_filename[70];
  sprintf(final_filename, "%s_prop_out.txt", SimParameters.output_file_name);
  if (remove(final_filename) != 0 || remove(final_filename) != 0) printf("Error deleting the old propagation files or it does not exist\n");
  else printf("Old propagation files deleted successfully\n");

  struct MonteCarloResult_t* Results = (struct MonteCarloResult_t*)malloc(SimParameters.NT*sizeof(MonteCarloResult_t));

  //..... Cycle on initial backward propafation energies .....//

  for (int iE=0; iE<SimParameters.NT; iE++) {

    // .. Initialize random generator
    curandStatePhilox4_32_10_t *dev_RndStates;
    cudaMalloc((void **)&dev_RndStates, SimParameters.Npart*sizeof(curandStatePhilox4_32_10_t));
    unsigned long Rnd_seed=getpid()+time(NULL)+iE;
    init_rdmgenerator<<<16, SimParameters.Npart/16>>>(dev_RndStates, Rnd_seed);
    cudaDeviceSynchronize();

    printf("\n-- Cycle on energy[%d]: %.2f \n", iE , SimParameters.Tcentr[iE]);

    float* host_outr = (float*)malloc(SimParameters.Npart*sizeof(float));
    float* host_outE = (float*)malloc(SimParameters.Npart*sizeof(float));
    float* host_alphapath = (float*)malloc(SimParameters.Npart*sizeof(float));

    float* temp_inr = (float*)malloc(SimParameters.Npart*sizeof(float));
    float* temp_inE = (float*)malloc(SimParameters.Npart*sizeof(float));

    for (int iPart=0; iPart<SimParameters.Npart; iPart++){
      int PeriodIndex = floor(iPart*SimParameters.NInitialPositions/SimParameters.Npart);
      host_outr[iPart] = SimParameters.InitialPosition[PeriodIndex];
      host_outE[iPart] = SimParameters.Tcentr[iE];
      host_alphapath[iPart] = 0;
      temp_inr[iPart] = host_outr[iPart];
      temp_inE[iPart] = host_outE[iPart];
    }

    // Device propagatiion variable initialization
    float* dev_outr;
    cudaMalloc((void **)&dev_outr, SimParameters.Npart*sizeof(float));
    cudaMemcpy(dev_outr, host_outr, SimParameters.Npart*sizeof(float), cudaMemcpyHostToDevice);
    float* dev_outE;
    cudaMalloc((void **)&dev_outE, SimParameters.Npart*sizeof(float));
    cudaMemcpy(dev_outE, host_outE, SimParameters.Npart*sizeof(float), cudaMemcpyHostToDevice);
    float* dev_alphapath;
    cudaMalloc((void **)&dev_alphapath, SimParameters.Npart*sizeof(float));
    cudaMemcpy(dev_alphapath, host_alphapath, SimParameters.Npart*sizeof(float), cudaMemcpyHostToDevice);

    // Propagation
    propagation<<<16, SimParameters.Npart/16>>>(dev_RndStates, dev_outr, dev_outE, dev_alphapath, SimParameters);
    cudaDeviceSynchronize();

    // printGPU<<<1,1>>>(dev_outE, SimParameters.Npart);

    cudaMemcpy(host_outr, dev_outr, SimParameters.Npart*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_outE, dev_outE, SimParameters.Npart*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_alphapath, dev_alphapath, SimParameters.Npart*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_outr);
    cudaFree(dev_outE);
    cudaFree(dev_alphapath);

    // Find out E max
    float Emax = findMax(host_outE, SimParameters.Npart);

    // Definition of histogram binning as a fraction of the bin border (DeltaT=T*RelativeBinAmplitude)
    float DeltaLogT= log10(1.+SimParameters.RelativeBinAmplitude);
    float LogBin0_lowEdge = log10(SimParameters.Tcentr[iE])-(DeltaLogT/2.);
    float Bin0_lowEdge = pow(10, LogBin0_lowEdge );                     // first LowEdge Bin

    Results[iE].Nbins           = ceilf(log10(Emax/Bin0_lowEdge)/DeltaLogT);
    Results[iE].LogBin0_lowEdge = LogBin0_lowEdge;
    Results[iE].DeltaLogT       = DeltaLogT;

    // Histogram building
    Results[iE].BoundaryDistribution = (float*)malloc(Results[iE].Nbins*sizeof(float));
    for (int iBin=0; iBin<Results[iE].Nbins; iBin++) {
      Results[iE].BoundaryDistribution[iBin] = 0;
    }
    // Failed quasi-particle propagation count
    int Nfailed=0;
    
    for (int iPart=0; iPart<SimParameters.Npart; iPart++) {
      if (log10(host_outE[iPart])>LogBin0_lowEdge) {
          int DestBin = floor( (log10(host_outE[iPart])-LogBin0_lowEdge)/DeltaLogT ); // evalaute the bin where put event
          Results[iE].BoundaryDistribution[DestBin] += exp(host_alphapath[iPart]);
          // Check histogrm filling (debug)
          if (DestBin<0 || DestBin>Results[iE].Nbins) printf("ERROR: DestBin[%d] = %f must be inside [0,Results[iR].Nbins]\n", iPart, DestBin);
      }

      else Nfailed += 1;
    }

    // Check histogrm filling (debug)
    for (int iBin=0; iBin<Results[iE].Nbins; iBin++) {
      if (Results[iE].BoundaryDistribution[iBin]<0 || Results[iE].BoundaryDistribution[iBin]>SimParameters.Npart) printf("ERROR: Bin[%d] = %f must be inside [0,Npart]\n", iBin, Results[iE].BoundaryDistribution[iBin]);
    }

    Results[iE].Nregistered = SimParameters.Npart-Nfailed;

    printf("-- Eventi computati : %lu \n", SimParameters.Npart);
    printf("-- Eventi falliti   : %d \n", Nfailed);
    printf("-- Eventi registrati: %lu \n", Results[iE].Nregistered);

    // Save prticle history
    SaveTxt_part(final_filename, SimParameters.Npart, temp_inr, temp_inE, host_outr, host_outE, host_alphapath);

    free(temp_inr);
    free(temp_inE);

    cudaFree(dev_RndStates);

    free(host_outr);
    free(host_outE);
    free(host_alphapath);
  }

  //..... Save propagation output and nergy histograms .....//

  FILE* pFile_Matrix=NULL;
  char RAWMatrix_name[2000];
  sprintf(RAWMatrix_name,"%s_matrix_%lu.dat", SimParameters.output_file_name, (unsigned long int)getpid());
  pFile_Matrix = fopen(RAWMatrix_name, "w");

  fprintf(pFile_Matrix, "# COSMICA \n");
  fprintf(pFile_Matrix, "# Number of Input energies;\n");
  fprintf(pFile_Matrix, "%d \n", SimParameters.NT);

  for (int itemp=0; itemp<SimParameters.NT; itemp++) {

    fprintf(pFile_Matrix,"######  Bin %d \n", itemp);
    fprintf(pFile_Matrix,"# Egen, Npart Gen., Npart Registered, Nbin output, log10(lower edge bin 0), Bin amplitude (in log scale)\n");

    fprintf(pFile_Matrix,"%f %lu %lu %d %f %f \n",SimParameters.Tcentr[itemp],
                                                  SimParameters.Npart,
                                                  Results[itemp].Nregistered,
                                                  Results[itemp].Nbins,
                                                  Results[itemp].LogBin0_lowEdge,
                                                  Results[itemp].DeltaLogT);

    fprintf(pFile_Matrix, "# output distribution \n");

    for (int itNB=0; itNB<Results[itemp].Nbins; itNB++) {
        fprintf(pFile_Matrix, "%e ", Results[itemp].BoundaryDistribution[itNB]);
    }

    fprintf(pFile_Matrix,"\n");
    fprintf(pFile_Matrix,"#\n"); // <--- dummy line to separate results
  }

  fflush(pFile_Matrix);
  fclose(pFile_Matrix);

  free(Results);
}
