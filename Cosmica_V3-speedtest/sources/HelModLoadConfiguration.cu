#include <stdio.h>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions
#include <errno.h>          // Defines the external errno variable and all the values it can take on
#include <unistd.h>         // Supplies EXIT_FAILURE, EXIT_SUCCESS
#include <libgen.h>         // Supplies the basename() function

#include "HelModLoadConfiguration.cuh"
#include "HelModVariableStructure.cuh"
#include "VariableStructure.cuh"
#include "VariableStructure.cuh"
#include "GenComputation.cuh"
#include "DiffusionModel.cuh"
#include "MagneticDrift.cuh"

// Define load function parameteres
#define ERR_Load_Configuration_File "Error while loading simulation parameters \n"
#define LOAD_CONF_FILE_SiFile "Configuration file loaded \n"
#define LOAD_CONF_FILE_NoFile "No configuration file Specified. default value used instead \n"
#define ERR_NoOutputFile "ERROR: output file cannot be open, do you have writing permission?\n"

// -----------------------------------------------------------------
// ------------------  External Declaration  -----------------------
// -----------------------------------------------------------------
extern int errno;
extern char *optarg;
extern int opterr, optind;
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
#define WELCOME "Welcome to COSMICA, enjoy the speedy side of propagation\n"
#define DEFAULT_PROGNAME "Cosmica"
#define OPTSTR "vi:h"
#define USAGE_MESSAGE "Thanks for using Cosmica, To execute this program please specify: "
#define USAGE_FMT  "%s [-v] -i <inputfile> [-h] \n"

void usage(char *progname, int opt) {
   fprintf(stderr, USAGE_MESSAGE); 
   fprintf(stderr, USAGE_FMT, progname?progname:DEFAULT_PROGNAME);
   exit(EXIT_FAILURE);
   /* NOTREACHED */
}

void kill_me(const char *REASON) {
    perror(REASON);
    exit(EXIT_FAILURE);
}

int PrintError (const char *var, char *value, int zone){
  fprintf (stderr,"ERROR: %s value not valid [actual value %s for region %d] \n",var,value,zone); 
  return EXIT_FAILURE; 
}

unsigned char SplitCSVString(const char *InputString, float **Outputarray)
{
  unsigned char Nelements=0;
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

int Load_Configuration_File(int argc, char* argv[], struct SimParameters_t &SimParameters, int verbose) {
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

      case 'v':
        verbose += 1;
        break;

      case 'h':
      default:
        usage(basename(argv[0]), opt);
        /* NOTREACHED */
        break;
    }
  if (verbose) { 
    printf(WELCOME);
    switch (verbose){
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
    if (verbose>=VERBOSE_med){
      fprintf(stderr,"-- --- Init ---\n");
      fprintf(stderr,"-- you entered %d arguments:\n",argc);
      for (int i = 0; i<argc; i++){ fprintf(stderr,"-->  %s \n",argv[i]);}
    }
  }

  //................. load simulation parameters 

  // .. check integrity
  if (!opt) {
    errno = EINVAL;
    kill_me(ERR_Load_Configuration_File);
  }

  if (!input) {
    errno = ENOENT;
    kill_me(ERR_Load_Configuration_File);
  }

  if (input== stdin){
  	kill_me("No configuration file Specified, Use -i option \n");
  }

  /** 
   * La lettura del file di configurazione avviene attraverso un ciclo riga per riga
   * Il file è strutturato in un configurazione chiave:valore quindi la lettura della riga
   * permette di indetificare la chiave e di modificare la variabile corrispondente 
   * nella struct SimParameters.
   **/


  // .. Use default flags
  bool UseDefault_initial_energy=true;

  // .. Heliospheric parameters
  int NumberParameterLoaded  = 0;  // count number of parameters loaded
  int NumberHeliosheatParLoaded =0 ; 
  unsigned char Nregions=0;
  InputHeliosphericParameters_t IHP[NMaxRegions]; //
  InputHeliosheatParameters_t   IHS[NMaxRegions];

  // .. initial positions
  float *r; 
  float *th;
  float *ph;
  unsigned char N_r=0;
  unsigned char N_th=0;
  unsigned char N_ph=0;

  // .. load Conf File
  if (input!= stdin){

    /**  inserire qui la lettura del file in input. 
     * le chiavi caricate modificano il contenuto di SimParameters
     * poi nella parte verbose si indica cosa è stato caricato dal file 
     **/
    char line[ReadingStringLenght];
    char key[ReadingStringLenght],value[ReadingStringLenght];
    while ((fgets(line, ReadingStringLenght, input)) != NULL)
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
        if (verbose>=VERBOSE_hig) { 
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
        if (verbose>=VERBOSE_hig) { 
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
        if (verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### number of input SourcePos_r from configuration file is %d\n",N_r);
          fprintf(stdout,"### SourcePos_r--> ");
          for (int i_split=0; i_split< N_r; i_split++){ fprintf(stdout,"%f ",r[i_split]);  }
          fprintf(stdout,"\n"); 
        }
      }

      if (strcmp(key,"SourcePos_theta")==0){
        N_th=SplitCSVString(value, &th);
        if (verbose>=VERBOSE_hig) { 
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
        if (verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### number of input SourcePos_phi from configuration file is %d\n",N_ph);
          fprintf(stdout,"### SourcePos_phi--> ");
          for (int i_split=0; i_split< N_ph; i_split++){ fprintf(stdout,"%f ",ph[i_split]);  }
          fprintf(stdout,"\n"); 
        }
      }      
      // ------------- particle description ----------------
      if (strcmp(key,"Particle_NucleonRestMass")==0){ 
        SimParameters.IonToBeSimulated.T0= atof(value);
        if (verbose>=VERBOSE_hig) { 
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
        if (verbose>=VERBOSE_hig) { 
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
        if (verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### Particle_Charge--> %s\n",value);
        }
      }

      // ------------- Number of region in which divide the heliosphere (Heliosheet excluded) ------
      if (strcmp(key,"Nregions")==0){ 
        Nregions= (unsigned char)atoi(value);
        if (verbose>=VERBOSE_hig) { 
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
        if (verbose>=VERBOSE_hig) { 
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
        if (verbose>=VERBOSE_hig) { 
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
        if (verbose>=VERBOSE_hig) { 
          fprintf(stdout,"### RelativeBinAmplitude--> %s\n",value);
        }
        if (SimParameters.RelativeBinAmplitude>0.01)
        {
          fprintf (stderr,"ERROR: RelativeBinAmplitude cannot be greater than 1%% (0.01) if you whish to have decent results. \n");
          return EXIT_FAILURE;
        }
      }
    }// ------------- END parsing ----------------



    if (verbose>=VERBOSE_low) { 
      fprintf(stderr,LOAD_CONF_FILE_SiFile);
    }
    fclose(input);
  }
  
  else{
    if (verbose>=VERBOSE_low) { 
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
    if ((ph[iPos]<0) || (ph[iPos]>2*Pi))
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
      SimParameters.prop_medium[izone].k0_paral[0] = IHP[izone].k0;
      SimParameters.prop_medium[izone].k0_paral[1] = IHP[izone].k0;
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
                        verbose);
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
                        verbose);
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
      if (verbose>=VERBOSE_hig) {fprintf(stdout,"### BIN::\t%d\t%.2f\n",j,SimParameters.Tcentr[j]);}
    }
  }



  // .. init other variables
  SimParameters.Results                 = (MonteCarloResult_t*)malloc( SimParameters.NT * sizeof(MonteCarloResult_t) );
 

  // .. recap simulation parameters  - print SimParameters_t content
  if (verbose>=VERBOSE_med) { 
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
