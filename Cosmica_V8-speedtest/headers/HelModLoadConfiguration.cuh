#ifndef HelModLoadConfiguration
#define HelModLoadConfiguration
#include <HelModVariableStructure.cuh>

struct InputHeliosphericParameters_t {
    float k0 = 0;
    float ssn = 0;
    float V0 = 0;
    float TiltAngle = 0;
    float SmoothTilt = 0;
    float BEarth = 0;
    int Polarity = 0;
    int SolarPhase = 0;
    float NMCR = 0;
    float Rts_nose = 0;
    float Rts_tail = 0;
    float Rhp_nose = 0;
    float Rhp_tail = 0;
};

void usage(const char *);

/* Loads program options
    */

void kill_me(const char *);

/* Kills the program for REASON
    */

int PrintError(const char *, char *, int);

/* Print error for value out of allowed range
    */


// int LoadConfigFile(int, char **, struct SimParameters_t &, int);
int LoadConfigFile(int, char *[], SimConfiguration_t &, int);

int LoadConfigYaml(int, char *[], SimConfiguration_t &, int);

/* Load the simulation global parameters from the configuration file
    NOTE: USING OLD STABLE 4_CoreCode_MultiGPU_MultiYear VERSION 
    */

int write_results_yaml(const char *, const SimConfiguration_t &);

#endif
