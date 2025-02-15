#include <cstdio>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions
#include <unistd.h>         // Supplies EXIT_FAILURE, EXIT_SUCCESS

#include "HelModLoadConfiguration.cuh"

#include <fstream>
#include <GenComputation.cuh>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "VariableStructure.cuh"
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
// extern int errno;
// extern char *optarg;
// extern int opterr, optind;
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
#define WELCOME "Welcome to COSMICA, enjoy the speedy side of propagation\n"
#define DEFAULT_PROGNAME "Cosmica"
#define OPTSTR "vi:h"
#define USAGE_MESSAGE "Thanks for using Cosmica, To execute this program please specify: "
#define USAGE_FMT  "%s [-v] -i <inputfile> [-h] \n"

using std::vector, std::string, std::unordered_map;

void usage(const char *progname) {
    fprintf(stderr, USAGE_MESSAGE);
    fprintf(stderr, USAGE_FMT, progname ? progname : DEFAULT_PROGNAME);
    exit(EXIT_FAILURE);
    /* NOTREACHED */
}

void kill_me(const char *REASON) {
    perror(REASON);
    exit(EXIT_FAILURE);
}

int PrintError(const char *var, char *value, const int zone) {
    fprintf(stderr, "ERROR: %s value not valid [actual value %s for region %d] \n", var, value, zone);
    return EXIT_FAILURE;
}


vector<float> SplitCSV(const string &str) {
    vector<float> tempFloats;
    std::stringstream ss(str);
    string token;
    constexpr char delimiter = ',';

    while (std::getline(ss, token, delimiter)) {
        try {
            tempFloats.push_back(std::stof(token));
        } catch (const std::invalid_argument &e) {
            std::cerr << "Invalid float value: " << token << " - " << e.what() << std::endl;
        } catch (const std::out_of_range &e) {
            std::cerr << "Float value out of range: " << token << " - " << e.what() << std::endl;
        }
    }

    return tempFloats;
}

InputHeliosphericParameters_t ParseHeliospheric(const string &str) {
    const auto parsed = SplitCSV(str);
    return {
        parsed[0],
        parsed[1],
        parsed[2],
        parsed[3],
        parsed[4],
        parsed[5],
        static_cast<int>(parsed[6]),
        static_cast<int>(parsed[7]),
        parsed[8],
        parsed[9],
        parsed[10],
        parsed[11],
        parsed[12],
    };
}

InputHeliosheatParameters_t ParseHeliosheat(const string &str) {
    const auto parsed = SplitCSV(str);
    return {parsed[0], parsed[1]};
}

unordered_map<string, string> ParseCLIArguments(const int argc, char *argv[]) {
    unordered_map<string, string> cliArgs;

    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cliArgs[string(argv[i]).substr(1)] = string(argv[i + 1]);
                i += 1;
            } else {
                cliArgs[string(argv[i]).substr(1)] = "true";
            }
        }
    }

    return cliArgs;
}

std::pair<string, string> splitKeyValue(const string &line) {
    const size_t colonPos = line.find(':');
    if (colonPos == string::npos) {
        throw std::invalid_argument("Invalid format. Expected 'key: value'");
    }
    // Extract key and value
    string key = line.substr(0, colonPos);
    string value = line.substr(colonPos + 1); // Skip the space after ':'

    // Trim leading/trailing whitespaces
    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);
    value.erase(0, value.find_first_not_of(" \t"));
    value.erase(value.find_last_not_of(" \t") + 1);

    return {key, value};
}

#ifndef UNIFIED_COMPILE
constexpr unsigned long hash(const std::string_view &str) {
     unsigned long hash = 0;
     for (const auto &e: str) hash = hash * 131 + e;
     return hash;
}

consteval unsigned long operator""_(const char *str, const size_t len) {
     return hash(std::string_view(str, len));
}
#endif

int LoadConfigFile(int argc, char *argv[], SimParameters_t &SimParameters, int verbose) {
    auto options = ParseCLIArguments(argc, argv);
    if (options.contains("v")) verbose += 1;
    else if (options.contains("vv")) verbose += 2;
    else if (options.contains("vvv")) verbose += 3;
    std::ifstream file(options["i"]);

    if (verbose) {
        printf(WELCOME);
        switch (verbose) {
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
        if (verbose >= VERBOSE_med) {
            fprintf(stderr, "-- --- Init ---\n");
            fprintf(stderr, "-- you entered %d arguments:\n", argc);
            for (int i = 0; i < argc; i++) { fprintf(stderr, "-->  %s \n", argv[i]); }
        }
    }

    vector<float> SPr, SPth, SPphi, Ts;
    vector<InputHeliosphericParameters_t> IHP;
    vector<InputHeliosheatParameters_t> IHS;

    string line;
    while (std::getline(file, line)) {
        if (line[0] == '#') continue;
        try {
            auto [key, value] = splitKeyValue(line);
            switch (hash(key)) {
                case "RandomSeed"_:
                    SimParameters.RandomSeed = std::stol(value);
                case "OutputFilename"_:
                    strncpy(SimParameters.output_file_name, value.c_str(), struct_string_lengh);
                    break;
                case "Tcentr"_:
                    Ts = SplitCSV(value);
                    SimParameters.NT = Ts.size();
                    SimParameters.Tcentr = new float[SimParameters.NT];
                    std::copy(Ts.begin(), Ts.end(), SimParameters.Tcentr);
                    break;
                case "Npart"_:
                    SimParameters.Npart = std::stoi(value);
                    break;
                case "SourcePos_r"_:
                    SPr = SplitCSV(value);
                    break;
                case "SourcePos_theta"_:
                    SPth = SplitCSV(value);
                    break;
                case "SourcePos_phi"_:
                    SPphi = SplitCSV(value);
                    break;
                case "Particle_NucleonRestMass"_:
                    SimParameters.IonToBeSimulated.T0 = std::stof(value);
                    break;
                case "Particle_MassNumber"_:
                    SimParameters.IonToBeSimulated.A = std::stof(value);
                    break;
                case "Particle_Charge"_:
                    SimParameters.IonToBeSimulated.Z = std::stof(value);
                    break;
                case "Nregions"_:
                    SimParameters.HeliosphereToBeSimulated.Nregions = std::stoi(value);
                    break;
                case "HeliosphericParameters"_:
                    IHP.push_back(ParseHeliospheric(value));
                    break;
                case "HeliosheatParameters"_:
                    IHS.push_back(ParseHeliosheat(value));
                    break;
                case "RelativeBinAmplitude"_:
                    SimParameters.RelativeBinAmplitude = std::stof(value);
                    break;
                default:
                    std::cout << "No parser found for key: " << key << ". Skipping..." << std::endl;
            }
        } catch (const std::exception &e) {
            std::cerr << "Error processing line: " << e.what() << std::endl;
        }
    }

    if (SPr.size() != SPth.size() || SPr.size() != SPphi.size()) {
        std::cerr << "Mismatched initial positions\n";
        return EXIT_FAILURE;
    }
    // if (SPr.size() + SimParameters.HeliosphereToBeSimulated.Nregions - 1 != IHP.size()) {
    //     std::cerr << "Mismatched initial positions and regions " << SPr.size() << ' ' << SimParameters.
    //             HeliosphereToBeSimulated.Nregions << ' ' << IHP.size() << std::endl;
    //     return EXIT_FAILURE;
    // }

    SimParameters.NInitialPositions = SPr.size();
    SimParameters.InitialPosition = new vect3D_t[SimParameters.NInitialPositions];

    for (size_t i = 0; i < SimParameters.NInitialPositions; ++i) {
        SimParameters.InitialPosition[i] = {SPr[i], SPth[i], SPphi[i]};

        float mean_tilt = 0;
        for (size_t j = i; j < SimParameters.HeliosphereToBeSimulated.Nregions + i; ++j) mean_tilt += IHP[j].TiltAngle;
        mean_tilt /= static_cast<float>(SimParameters.HeliosphereToBeSimulated.Nregions);

        SimParameters.HeliosphereToBeSimulated.IsHighActivityPeriod[i] = mean_tilt >= TiltL_MaxActivity_threshold;
        SimParameters.HeliosphereToBeSimulated.RadBoundary_real[i].Rts_nose = IHP[i].Rts_nose;
        SimParameters.HeliosphereToBeSimulated.RadBoundary_real[i].Rts_tail = IHP[i].Rts_tail;
        SimParameters.HeliosphereToBeSimulated.RadBoundary_real[i].Rhp_nose = IHP[i].Rhp_nose;
        SimParameters.HeliosphereToBeSimulated.RadBoundary_real[i].Rhp_tail = IHP[i].Rhp_tail;
    }

    for (size_t i = 0; i < IHP.size(); ++i) {
        SimParameters.prop_medium[i].V0 = IHP[i].V0 / aukm;
        if (IHP[i].k0 > 0) {
            SimParameters.prop_medium[i].k0_paral[0] = IHP[i].k0;
            SimParameters.prop_medium[i].k0_paral[1] = IHP[i].k0;
            SimParameters.prop_medium[i].k0_perp[0] = IHP[i].k0;
            SimParameters.prop_medium[i].k0_perp[1] = IHP[i].k0;
            SimParameters.prop_medium[i].GaussVar[0] = 0;
            SimParameters.prop_medium[i].GaussVar[1] = 0;
        } else {
            auto [kxh, kyh,kzh] = EvalK0(true, // isHighActivity
                                         IHP[i].Polarity, SimParameters.IonToBeSimulated.Z, IHP[i].SolarPhase,
                                         IHP[i].SmoothTilt, IHP[i].NMCR, IHP[i].ssn, verbose);
            auto [kxl, kyl,kzl] = EvalK0(true, // isHighActivity
                                         IHP[i].Polarity, SimParameters.IonToBeSimulated.Z, IHP[i].SolarPhase,
                                         IHP[i].SmoothTilt, IHP[i].NMCR, IHP[i].ssn, verbose);
            SimParameters.prop_medium[i].k0_paral[0] = kxh;
            SimParameters.prop_medium[i].k0_perp[0] = kyh;
            SimParameters.prop_medium[i].GaussVar[0] = kzh;
            SimParameters.prop_medium[i].k0_paral[1] = kxl;
            SimParameters.prop_medium[i].k0_perp[1] = kyl;
            SimParameters.prop_medium[i].GaussVar[1] = kzl;
        }
        SimParameters.prop_medium[i].g_low = g_low(IHP[i].SolarPhase, IHP[i].Polarity, IHP[i].SmoothTilt);
        SimParameters.prop_medium[i].rconst = rconst(IHP[i].SolarPhase, IHP[i].Polarity, IHP[i].SmoothTilt);
        SimParameters.prop_medium[i].TiltAngle = IHP[i].TiltAngle * Pi / 180.f; // conversion to radian
        SimParameters.prop_medium[i].Asun = static_cast<float>(IHP[i].Polarity) * sq(aum) * IHP[i].BEarth * 1e-9f /
                                            sqrtf(1.f + Omega * (1 - rhelio) / (IHP[i].V0 / aukm) * (
                                                      Omega * (1 - rhelio) / (IHP[i].V0 / aukm)));

        //HelMod-Like
        SimParameters.prop_medium[i].P0d = EvalP0DriftSuppressionFactor(0, IHP[i].SolarPhase, IHP[i].TiltAngle, 0);
        SimParameters.prop_medium[i].P0dNS = EvalP0DriftSuppressionFactor(
            1, IHP[i].SolarPhase, IHP[i].TiltAngle, IHP[i].ssn);
        SimParameters.prop_medium[i].plateau = EvalHighRigidityDriftSuppression_plateau(
            IHP[i].SolarPhase, IHP[i].TiltAngle);
    }

    for (int i = 0; i < IHS.size(); i++) {
        SimParameters.prop_Heliosheat[i].k0 = IHS[i].k0;
        SimParameters.prop_Heliosheat[i].V0 = IHS[i].V0 / aukm;
    }

    SimParameters.Results = new MonteCarloResult_t[SimParameters.NT];

    if (verbose >= VERBOSE_med) {
        fprintf(stderr, "----- Recap of Simulation parameters ----\n");
        fprintf(stderr, "NucleonRestMass         : %.3f Gev/n \n", SimParameters.IonToBeSimulated.T0);
        fprintf(stderr, "MassNumber              : %.1f \n", SimParameters.IonToBeSimulated.A);
        fprintf(stderr, "Charge                  : %.1f \n", SimParameters.IonToBeSimulated.Z);
        fprintf(stderr, "Number of sources       : %hhu \n", SimParameters.NInitialPositions);
        for (int i = 0; i < SimParameters.NInitialPositions; i++) {
            fprintf(stderr, "position              :%d \n", i);
            fprintf(stderr, "  Init Pos (real) - r     : %.2f \n", SimParameters.InitialPosition[i].r);
            fprintf(stderr, "  Init Pos (real) - theta : %.2f \n", SimParameters.InitialPosition[i].th);
            fprintf(stderr, "  Init Pos (real) - phi   : %.2f \n", SimParameters.InitialPosition[i].phi);
        }
        fprintf(stderr, "output_file_name        : %s \n", SimParameters.output_file_name);
        fprintf(stderr, "number of input energies: %d \n", SimParameters.NT);
        fprintf(stderr, "input energies          : ");
        for (int i = 0; i < SimParameters.NT; i++) {
            fprintf(stderr, "%.2f ", SimParameters.Tcentr[i]);
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "Events to be generated  : %lu \n", SimParameters.Npart);
        //fprintf(stderr,"Warp per Block          : %d \n",WarpPerBlock);

        fprintf(stderr, "\n");
        fprintf(stderr, "for each simulated periods:\n");
        for (int i = 0; i < SimParameters.NInitialPositions; i++) {
            fprintf(stderr, "position              :%d \n", i);
            fprintf(stderr, "  IsHighActivityPeriod    : %s \n",
                    SimParameters.HeliosphereToBeSimulated.IsHighActivityPeriod[i] ? "true" : "false");
            fprintf(stderr, "  Rts nose direction      : %.2f AU\n",
                    SimParameters.HeliosphereToBeSimulated.RadBoundary_real[i].Rts_nose);
            fprintf(stderr, "  Rts tail direction      : %.2f AU\n",
                    SimParameters.HeliosphereToBeSimulated.RadBoundary_real[i].Rts_tail);
            fprintf(stderr, "  Rhp nose direction      : %.2f AU\n",
                    SimParameters.HeliosphereToBeSimulated.RadBoundary_real[i].Rhp_nose);
            fprintf(stderr, "  Rhp tail direction      : %.2f AU\n",
                    SimParameters.HeliosphereToBeSimulated.RadBoundary_real[i].Rhp_tail);
        }
        fprintf(stderr, "Heliopshere Parameters ( %d regions ): \n", SimParameters.HeliosphereToBeSimulated.Nregions);

        for (int i = 0; i < SimParameters.HeliosphereToBeSimulated.Nregions + SimParameters.
                        NInitialPositions - 1; i++) {
            fprintf(stderr, "- Region %d \n", i);
            fprintf(stderr, "-- V0         %e AU/s\n", SimParameters.prop_medium[i].V0);
            fprintf(stderr, "-- k0_paral   [%e,%e] \n", SimParameters.prop_medium[i].k0_paral[0],
                    SimParameters.prop_medium[i].k0_paral[1]);
            fprintf(stderr, "-- k0_perp    [%e,%e] \n", SimParameters.prop_medium[i].k0_perp[0],
                    SimParameters.prop_medium[i].k0_perp[1]);
            fprintf(stderr, "-- GaussVar   [%.4f,%.4f] \n", SimParameters.prop_medium[i].GaussVar[0],
                    SimParameters.prop_medium[i].GaussVar[1]);
            fprintf(stderr, "-- g_low      %.4f \n", SimParameters.prop_medium[i].g_low);
            fprintf(stderr, "-- rconst     %.3f \n", SimParameters.prop_medium[i].rconst);
            fprintf(stderr, "-- tilt angle %.3f rad\n", SimParameters.prop_medium[i].TiltAngle);
            fprintf(stderr, "-- Asun       %e \n", SimParameters.prop_medium[i].Asun);
            fprintf(stderr, "-- P0d        %e GV \n", SimParameters.prop_medium[i].P0d);
            fprintf(stderr, "-- P0dNS      %e GV \n", SimParameters.prop_medium[i].P0dNS);


            // XXXXXXX
        }
        fprintf(stderr, "Heliosheat parameters ( %d periods ): \n", SimParameters.NInitialPositions);
        for (int ipos = 0; ipos < SimParameters.NInitialPositions; ipos++) {
            fprintf(stderr, "-period              :%d \n", ipos);
            fprintf(stderr, "-- V0 %e AU/s\n", SimParameters.prop_Heliosheat[ipos].V0);
            fprintf(stderr, "-- k0 %e \n", SimParameters.prop_Heliosheat[ipos].k0);
        }
        fprintf(stderr, "----------------------------------------\n");
    }

    return EXIT_SUCCESS;
}
