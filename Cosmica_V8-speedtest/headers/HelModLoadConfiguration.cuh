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

[[noreturn]] void usage(const char *);

void kill_me(const char *);

int PrintError(const char *, char *, int);

int LoadConfigFile(int, char *[], SimConfiguration_t &, int);

int LoadConfigYaml(int, char *[], SimConfiguration_t &, int);

int write_results_yaml(const std::string &, const SimConfiguration_t &);

#endif
