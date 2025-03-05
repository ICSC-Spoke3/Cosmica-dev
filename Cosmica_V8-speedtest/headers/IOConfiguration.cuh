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
    std::string input_file{}, output_dir{}, log_file{};
    bool use_stdin = false, use_stdout = false;
    spdlog::level::level_enum log_level = spdlog::level::info;
};


cli_options parse_cli_options(int, const char **);

void kill_me(const char *);

int PrintError(const char *, char *, int);

int LoadConfigFile(const cli_options &, SimConfiguration_t &);

int LoadConfigTxt(std::istream *, const cli_options &, SimConfiguration_t &);

int LoadConfigYaml(std::istream *, const cli_options &, SimConfiguration_t &);

std::string write_results_yaml(const cli_options &, const SimConfiguration_t &);

#endif
