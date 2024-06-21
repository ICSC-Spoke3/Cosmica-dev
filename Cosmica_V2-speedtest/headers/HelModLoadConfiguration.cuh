#ifndef HelModLoadConfiguration
#define HelModLoadConfiguration

void usage(char*, int);
/* Loads program options
    */

void kill_me(const char*);
/* Kills the program for REASON
    */

int PrintError (const char*, char*, int);
/* Print error for value out of allowed range
    */

unsigned char SplitCSVString(const char*, float**);
/* Split comma separated value
    */

int Load_Configuration_File(int, char**, struct SimParameters_t &, int);
/* Load the simulation global parameters from the configuration file
    NOTE: USING OLD STABLE 4_CoreCode_MultiGPU_MultiYear VERSION 
    */

#endif