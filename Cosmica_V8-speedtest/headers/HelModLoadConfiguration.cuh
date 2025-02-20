#ifndef HelModLoadConfiguration
#define HelModLoadConfiguration
#include <HelModVariableStructure.cuh>

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
int LoadConfigFile(int argc, char *argv[], SimParameters_t &SimParameters, int verbose);

int LoadConfigYaml(int argc, char *argv[], SimParameters_t &SimParameters, int verbose);

/* Load the simulation global parameters from the configuration file
    NOTE: USING OLD STABLE 4_CoreCode_MultiGPU_MultiYear VERSION 
    */

#endif
