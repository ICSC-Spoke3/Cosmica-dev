#ifndef HelModLoadConfiguration
#define HelModLoadConfiguration
#include <HelModVariableStructure.cuh>
#include <lyra/lyra.hpp>

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

struct cli_options {
    std::string input_file;
    std::string output_dir;
    spdlog::level::level_enum log_level = spdlog::level::info;
};


cli_options parse_cli_options(int, const char **);

void kill_me(const char *);

int PrintError(const char *, char *, int);

int LoadConfigFile(int, char *[], SimConfiguration_t &, int);

int LoadConfigYaml(const cli_options&, SimConfiguration_t &, int);

int write_results_yaml(const std::string &, const SimConfiguration_t &);

#endif
