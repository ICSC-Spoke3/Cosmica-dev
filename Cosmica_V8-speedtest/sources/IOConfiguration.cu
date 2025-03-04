#include <cstdio>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions

#include "IOConfiguration.cuh"

#include <fstream>
#include <GenComputation.cuh>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "VariableStructure.cuh"
#include "DiffusionModel.cuh"
#include "MagneticDrift.cuh"
#include <fkYAML.hpp>
#include <LoadConfiguration.cuh>
// Define load function parameteres
#define ERR_Load_Configuration_File "Error while loading simulation parameters \n"
#define LOAD_CONF_FILE_SiFile "Configuration file loaded \n"
#define LOAD_CONF_FILE_NoFile "No configuration file Specified. default value used instead \n"
#define ERR_NoOutputFile "ERROR: output file cannot be open, do you have writing permission?\n"


#define WELCOME "Welcome to COSMICA, enjoy the speedy side of propagation\n"
#define DEFAULT_PROGNAME "Cosmica"
#define OPTSTR "vi:h"
#define USAGE_MESSAGE "Thanks for using Cosmica, To execute this program please specify: "
#define USAGE_FMT  "%s [-v] -i <inputfile> [-h] \n"

using std::vector, std::string, std::pair, std::unordered_map;

/**
 * @brief Print the usage message
 * @param progname the name of the program
 *
 * @note noreturn: the function will never return a value
 */
[[noreturn]] void usage(const char *progname) {
    fprintf(stderr, USAGE_MESSAGE);
    fprintf(stderr, USAGE_FMT, progname ? progname : DEFAULT_PROGNAME);
    exit(EXIT_FAILURE);
}

/**
 * @brief Kill the program with a message
 * @param REASON the message to print
 */
void kill_me(const char *REASON) {
    perror(REASON);
    exit(EXIT_FAILURE);
}

/**
 * @brief Print an error message
 * @param var the variable name
 * @param value the value that caused the error
 * @param zone the region where the error occurred
 * @return EXIT_FAILURE
 */
int PrintError(const char *var, char *value, const int zone) {
    fprintf(stderr, "ERROR: %s value not valid [actual value %s for region %d] \n", var, value, zone);
    return EXIT_FAILURE;
}

/**
 * @brief Split a CSV string into a vector of floats using as delimiter ','
 * @param str the string to split
 * @return a vector of floats from the CSV string
 */
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

/**
 * @brief Parse a CSV string into a InputHeliosphericParameters_t struct
 * @param str the string to parse
 * @return a InputHeliosphericParameters_t struct
 */
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

/**
 * @brief Parse a CSV string into a InputHeliosheatParameters_t struct
 * @param str the string to parse
 * @return a InputHeliosheatParameters_t struct
 */
InputHeliosheatParameters_t ParseHeliosheat(const string &str) {
    const auto parsed = SplitCSV(str);
    return {parsed[0], parsed[1]};
}

/**
 * @brief Parse the command line arguments
 * @param argc the number of arguments
 * @param argv the arguments
 * @return a map of the arguments
 */
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

/**
 * @brief Split a string into a key-value pair
 * @param line the string to split
 * @return a pair of strings with the key and value
 *
 * @throws std::invalid_argument if the string is not in the expected format
 */
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

/**
 * @brief Load the configuration file
 * @param argc the number of arguments
 * @param argv the arguments
 * @param SimParameters the simulation parameters
 * @param verbose the verbosity level
 * @return EXIT_SUCCESS if the configuration was loaded successfully, EXIT_FAILURE otherwise
 */
int LoadConfigFile(int argc, char *argv[], SimConfiguration_t &SimParameters, int verbose) {
    auto options = ParseCLIArguments(argc, argv);

    if (options["i"].ends_with(".yaml")) return LoadConfigYaml(argc, argv, SimParameters, verbose);

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
            for (int i = 0; i < argc; ++i) { fprintf(stderr, "-->  %s \n", argv[i]); }
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
                    break;
                case "OutputFilename"_:
                    SimParameters.output_file_name = value;
                    break;
                case "Tcentr"_:
                    Ts = SplitCSV(value);
                    SimParameters.NT = Ts.size();
                    SimParameters.Tcentr = new float[SimParameters.NT];
                    std::ranges::copy(Ts, SimParameters.Tcentr);
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
                    SimParameters.simulation_constants.NIsotopes = 1;
                    SimParameters.simulation_constants.Isotopes[0].T0 = std::stof(value);
                    break;
                case "Particle_MassNumber"_:
                    SimParameters.simulation_constants.NIsotopes = 1;
                    SimParameters.simulation_constants.Isotopes[0].A = std::stof(value);
                    break;
                case "Particle_Charge"_:
                    SimParameters.simulation_constants.NIsotopes = 1;
                    SimParameters.simulation_constants.Isotopes[0].Z = std::stof(value);
                    break;
                case "Nregions"_:
                    SimParameters.simulation_constants.Nregions = std::stoi(value);
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

    SimParameters.NInitialPositions = SPr.size();
    std::ranges::copy(SPr, SimParameters.InitialPositions.r = new float[SPr.size()]);
    std::ranges::copy(SPth, SimParameters.InitialPositions.th = new float[SPth.size()]);
    std::ranges::copy(SPphi, SimParameters.InitialPositions.phi = new float[SPphi.size()]);

    for (size_t i = 0; i < SimParameters.NInitialPositions; ++i) {
        float mean_tilt = 0;
        for (size_t j = i; j < SimParameters.simulation_constants.Nregions + i; ++j) mean_tilt += IHP[j].TiltAngle;
        mean_tilt /= static_cast<float>(SimParameters.simulation_constants.Nregions);

        SimParameters.simulation_constants.IsHighActivityPeriod[i] = mean_tilt >= TiltL_MaxActivity_threshold;
        SimParameters.simulation_constants.RadBoundary_real[i].Rts_nose = IHP[i].Rts_nose;
        SimParameters.simulation_constants.RadBoundary_real[i].Rts_tail = IHP[i].Rts_tail;
        SimParameters.simulation_constants.RadBoundary_real[i].Rhp_nose = IHP[i].Rhp_nose;
        SimParameters.simulation_constants.RadBoundary_real[i].Rhp_tail = IHP[i].Rhp_tail;
    }

    std::ranges::copy(SimParameters.simulation_constants.RadBoundary_real,
                      SimParameters.simulation_constants.RadBoundary_effe);
    for (unsigned i = 0; i < SimParameters.NInitialPositions; ++i) {
        RescaleToEffectiveHeliosphere(SimParameters.simulation_constants.RadBoundary_effe[i],
                                      SimParameters.InitialPositions, i);
    }

    SimParameters.simulation_parametrization.Nparams = 1;
    SimParameters.simulation_parametrization.params = AllocateManaged<SimulationParametrizations_t::Parametrization_t
        []>(1);
    for (size_t i = 0; i < IHP.size(); ++i) {
        SimParameters.simulation_constants.heliosphere_properties[i].V0 = IHP[i].V0 / aukm;
        if (IHP[i].k0 > 0) {
            SimParameters.simulation_parametrization.params[0].heliosphere[i].k0_paral[0] = IHP[i].k0;
            SimParameters.simulation_parametrization.params[0].heliosphere[i].k0_paral[1] = IHP[i].k0;
            SimParameters.simulation_parametrization.params[0].heliosphere[i].k0_perp[0] = IHP[i].k0;
            SimParameters.simulation_parametrization.params[0].heliosphere[i].k0_perp[1] = IHP[i].k0;
            SimParameters.simulation_parametrization.params[0].heliosphere[i].GaussVar[0] = 0;
            SimParameters.simulation_parametrization.params[0].heliosphere[i].GaussVar[1] = 0;
        } else {
            auto [kxh, kyh,kzh] = EvalK0(true, // isHighActivity
                                         IHP[i].Polarity, SimParameters.simulation_constants.Isotopes[0].Z,
                                         IHP[i].SolarPhase, IHP[i].SmoothTilt, IHP[i].NMCR, IHP[i].ssn, verbose);
            auto [kxl, kyl,kzl] = EvalK0(false, // isHighActivity
                                         IHP[i].Polarity, SimParameters.simulation_constants.Isotopes[0].Z,
                                         IHP[i].SolarPhase, IHP[i].SmoothTilt, IHP[i].NMCR, IHP[i].ssn, verbose);
            SimParameters.simulation_parametrization.params[0].heliosphere[i].k0_paral[0] = kxh;
            SimParameters.simulation_parametrization.params[0].heliosphere[i].k0_perp[0] = kyh;
            SimParameters.simulation_parametrization.params[0].heliosphere[i].GaussVar[0] = kzh;
            SimParameters.simulation_parametrization.params[0].heliosphere[i].k0_paral[1] = kxl;
            SimParameters.simulation_parametrization.params[0].heliosphere[i].k0_perp[1] = kyl;
            SimParameters.simulation_parametrization.params[0].heliosphere[i].GaussVar[1] = kzl;
        }
        SimParameters.simulation_constants.heliosphere_properties[i].g_low = g_low(
            IHP[i].SolarPhase, IHP[i].Polarity, IHP[i].SmoothTilt);
        SimParameters.simulation_constants.heliosphere_properties[i].rconst = rconst(
            IHP[i].SolarPhase, IHP[i].Polarity, IHP[i].SmoothTilt);
        // Convert tilt angle to radians
        SimParameters.simulation_constants.heliosphere_properties[i].TiltAngle = IHP[i].TiltAngle * Pi / 180.f;
        // Calculate Asun
        SimParameters.simulation_constants.heliosphere_properties[i].Asun =
                static_cast<float>(IHP[i].Polarity) * sq(aum) * IHP[i].BEarth * 1e-9f /
                sqrtf(1.f + Omega * (1 - rhelio) / (IHP[i].V0 / aukm) * (
                          Omega * (1 - rhelio) / (IHP[i].V0 / aukm)));

        //HelMod-Like
        SimParameters.simulation_constants.heliosphere_properties[i].P0d = EvalP0DriftSuppressionFactor(
            0, IHP[i].SolarPhase, IHP[i].TiltAngle, 0);
        SimParameters.simulation_constants.heliosphere_properties[i].P0dNS = EvalP0DriftSuppressionFactor(
            1, IHP[i].SolarPhase, IHP[i].TiltAngle, IHP[i].ssn);
        SimParameters.simulation_constants.heliosphere_properties[i].plateau = EvalHighRigidityDriftSuppression_plateau(
            IHP[i].SolarPhase, IHP[i].TiltAngle);
    }

    for (unsigned i = 0; i < IHS.size(); ++i) {
        SimParameters.simulation_constants.heliosheat_properties[i].k0 = IHS[i].k0;
        SimParameters.simulation_constants.heliosheat_properties[i].V0 = IHS[i].V0 / aukm;
    }

    //TODO: initialize correctly
    // SimParameters.Results = new MonteCarloResult_t[SimParameters.NT];

    SimParameters.Npart = ceil_int_div(SimParameters.Npart, SimParameters.NInitialPositions);

    if (verbose >= VERBOSE_med) {
        fprintf(stderr, "----- Recap of Simulation parameters ----\n");
        fprintf(stderr, "NucleonRestMass         : %.3f Gev/n \n",
                SimParameters.simulation_constants.Isotopes[0].T0);
        fprintf(stderr, "MassNumber              : %.1f \n", SimParameters.simulation_constants.Isotopes[0].A);
        fprintf(stderr, "Charge                  : %.1f \n", SimParameters.simulation_constants.Isotopes[0].Z);
        fprintf(stderr, "Number of sources       : %hhu \n", SimParameters.NInitialPositions);
        for (unsigned i = 0; i < SimParameters.NInitialPositions; ++i) {
            fprintf(stderr, "position              :%d \n", i);
            fprintf(stderr, "  Init Pos (real) - r     : %.2f \n", SimParameters.InitialPositions.r[i]);
            fprintf(stderr, "  Init Pos (real) - theta : %.2f \n", SimParameters.InitialPositions.th[i]);
            fprintf(stderr, "  Init Pos (real) - phi   : %.2f \n", SimParameters.InitialPositions.phi[i]);
        }
        fprintf(stderr, "output_file_name        : %s \n", SimParameters.output_file_name.c_str());
        fprintf(stderr, "number of input energies: %d \n", SimParameters.NT);
        fprintf(stderr, "input energies          : ");
        for (unsigned i = 0; i < SimParameters.NT; ++i) {
            fprintf(stderr, "%.2f ", SimParameters.Tcentr[i]);
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "Events to be generated  : %u \n", SimParameters.Npart);
        //fprintf(stderr,"Warp per Block          : %d \n",WarpPerBlock);

        fprintf(stderr, "\n");
        fprintf(stderr, "for each simulated periods:\n");
        for (unsigned i = 0; i < SimParameters.NInitialPositions; ++i) {
            fprintf(stderr, "position              :%d \n", i);
            fprintf(stderr, "  IsHighActivityPeriod    : %s \n",
                    SimParameters.simulation_constants.IsHighActivityPeriod[i] ? "true" : "false");
            fprintf(stderr, "  Rts nose direction      : %.2f AU\n",
                    SimParameters.simulation_constants.RadBoundary_real[i].Rts_nose);
            fprintf(stderr, "  Rts tail direction      : %.2f AU\n",
                    SimParameters.simulation_constants.RadBoundary_real[i].Rts_tail);
            fprintf(stderr, "  Rhp nose direction      : %.2f AU\n",
                    SimParameters.simulation_constants.RadBoundary_real[i].Rhp_nose);
            fprintf(stderr, "  Rhp tail direction      : %.2f AU\n",
                    SimParameters.simulation_constants.RadBoundary_real[i].Rhp_tail);
        }
        fprintf(stderr, "Heliosphere Parameters ( %d regions ): \n", SimParameters.simulation_constants.Nregions);

        for (unsigned i = 0; i < SimParameters.simulation_constants.Nregions + SimParameters.NInitialPositions - 1; ++
             i) {
            fprintf(stderr, "- Region %d \n", i);
            fprintf(stderr, "-- V0         %e AU/s\n", SimParameters.simulation_constants.heliosphere_properties[i].V0);
            fprintf(stderr, "-- k0_paral   [%e,%e] \n",
                    SimParameters.simulation_parametrization.params[0].heliosphere[i].k0_paral[0],
                    SimParameters.simulation_parametrization.params[0].heliosphere[i].k0_paral[1]);
            fprintf(stderr, "-- k0_perp    [%e,%e] \n",
                    SimParameters.simulation_parametrization.params[0].heliosphere[i].k0_perp[0],
                    SimParameters.simulation_parametrization.params[0].heliosphere[i].k0_perp[1]);
            fprintf(stderr, "-- GaussVar   [%.4f,%.4f] \n",
                    SimParameters.simulation_parametrization.params[0].heliosphere[i].GaussVar[0],
                    SimParameters.simulation_parametrization.params[0].heliosphere[i].GaussVar[1]);
            fprintf(stderr, "-- g_low      %.4f \n",
                    SimParameters.simulation_constants.heliosphere_properties[i].g_low);
            fprintf(stderr, "-- rconst     %.3f \n",
                    SimParameters.simulation_constants.heliosphere_properties[i].rconst);
            fprintf(stderr, "-- tilt angle %.3f rad\n",
                    SimParameters.simulation_constants.heliosphere_properties[i].TiltAngle);
            fprintf(stderr, "-- Asun       %e \n", SimParameters.simulation_constants.heliosphere_properties[i].Asun);
            fprintf(stderr, "-- P0d        %e GV \n", SimParameters.simulation_constants.heliosphere_properties[i].P0d);
            fprintf(stderr, "-- P0dNS      %e GV \n",
                    SimParameters.simulation_constants.heliosphere_properties[i].P0dNS);
        }
        fprintf(stderr, "Heliosheat parameters ( %d periods ): \n", SimParameters.NInitialPositions);
        for (unsigned ipos = 0; ipos < SimParameters.NInitialPositions; ++ipos) {
            fprintf(stderr, "-period              :%d \n", ipos);
            fprintf(stderr, "-- V0 %e AU/s\n", SimParameters.simulation_constants.heliosheat_properties[ipos].V0);
            fprintf(stderr, "-- k0 %e \n", SimParameters.simulation_constants.heliosheat_properties[ipos].k0);
        }
        fprintf(stderr, "----------------------------------------\n");
    }

    return EXIT_SUCCESS;
}

/**
 * @brief From a YAML node, extract the particle description
 * @param node the YAML node
 * @param particle the particle description
 */
void from_node(const fkyaml::node &node, PartDescription_t &particle) {
    particle.T0 = node["nucleon_rest_mass"].get_value<float>();
    particle.A = node["mass_number"].get_value<float>();
    particle.Z = node["charge"].get_value<float>();
}

/**
 * @brief Extract the value from a YAML node
 * @param node the YAML node
 * @return the value
 */
template<typename T>
T node_to_value(const fkyaml::node &node) {
    return node.get_value<T>();
}

/**
 * @brief Extract a vector from a YAML node
 * @param node the YAML node
 * @return the vector
 */
template<typename T>
auto node_to_vector(const fkyaml::node &node) {
    vector<T> ret;
    std::ranges::transform(node, std::back_inserter(ret),
                           [](const auto &p) { return node_to_value<T>(p); });
    return ret;
}

/**
 * @brief Extract a map from a YAML node
 * @param node the YAML node
 * @return the map
 */
template<typename T>
auto node_to_map(const fkyaml::node &node) {
    pair<vector<string>, vector<T> > ret;
    std::ranges::transform(node.as_map(), std::back_inserter(ret.first),
                           [](const auto &p) { return p.first.as_str(); });
    std::ranges::transform(node.as_map(), std::back_inserter(ret.second),
                           [](const auto &p) { return node_to_value<T>(p.second); });
    return ret;
}

/**
 * @brief Check the number of sources
 * @param r the radial coordinates
 * @param th the polar coordinates
 * @param phi the azimuthal coordinates
 * @return the number of sources or exit if the coordinates are misaligned
 */
unsigned check_sources_count(const vector<float> &r, const vector<float> &th, const vector<float> &phi) {
    if (r.size() != th.size() || th.size() != phi.size()) {
        printf("Misaligned source coordinates (%lu, %lu, %lu)", r.size(), th.size(), phi.size());
        exit(EXIT_FAILURE);
    }
    return r.size();
}

/**
 * @brief Check the number of parameters
 * @param dynamic_node the dynamic node
 * @return the number of parameters
 */
unsigned check_params_count(const fkyaml::node &dynamic_node) {
    return dynamic_node["heliosphere"].cbegin().value().size();
}

/**
 * @brief Check the number of regions
 * @param isotopes the isotopes descriptions
 * @return the number of regions or exit if the isotopes have different charges
 */
int check_isotopes_charge(const vector<PartDescription_t> &isotopes) {
    if (isotopes.empty() || !std::ranges::all_of(isotopes, [&](const auto &i) { return i.Z == isotopes.front().Z; })) {
        std::cerr << "All isotopes must have the same charge" << std::endl;
        exit(EXIT_FAILURE);
    }
    return static_cast<int>(isotopes.front().Z);
}

/**
 * @brief Extract the heliosphere parameters from a YAML node
 * @param node the YAML node
 * @param n_sources the number of sources
 * @param n_regions the number of regions
 * @param n_param the number of parameters
 * @param z the charge of the isotopes
 * @return a tuple with the heliosphere parameters (heliosphere parametrization properties, heliosphere properties,
 *              high activity, boundary)
 */
std::tuple<vector<vector<HeliosphereParametrizationProperties_t> >, vector<HeliosphereProperties_t>, vector<bool>,
    vector<HeliosphereBoundRadius_t> >
node_to_heliosphere(const fkyaml::node &node, const unsigned n_sources, const unsigned n_regions,
                    const unsigned n_param, const int z) {
    const auto &static_node = node["static"]["heliosphere"], &dynamic_node = node["dynamic"]["heliosphere"];
    vector<vector<HeliosphereParametrizationProperties_t> > hpps(n_param);
    vector<HeliosphereProperties_t> hps;
    vector<bool> high_activity;
    vector<HeliosphereBoundRadius_t> boundary;
    for (unsigned i = 0; i < n_sources + n_regions - 1; ++i) {
        const InputHeliosphericProperties_t ihp{
            node_to_value<float>(static_node["ssn"][i]),
            node_to_value<float>(static_node["v0"][i]),
            node_to_value<float>(static_node["tilt_angle"][i]),
            node_to_value<float>(static_node["smooth_tilt"][i]),
            node_to_value<float>(static_node["b_field"][i]),
            node_to_value<int>(static_node["polarity"][i]),
            node_to_value<int>(static_node["solar_phase"][i]),
            node_to_value<float>(static_node["nmcr"][i]),
            node_to_value<float>(static_node["ts_nose"][i]),
            node_to_value<float>(static_node["ts_tail"][i]),
            node_to_value<float>(static_node["hp_nose"][i]),
            node_to_value<float>(static_node["hp_tail"][i]),
        };
        for (unsigned j = 0; j < n_param; ++j) {
            const InputHeliosphericParametrizationProperties_t ihpp{
                node_to_value<float>(dynamic_node["k0"][j][i]),
            };
            float kxh, kyh, kzh, kxl, kyl, kzl;
            if (ihpp.k0 > 0) {
                kxh = kxl = ihpp.k0;
                kyh = kyl = ihpp.k0;
                kzh = kzl = 0;
            } else {
                std::tie(kxh, kyh, kzh) = EvalK0(true, ihp.Polarity, z, ihp.SolarPhase, ihp.SmoothTilt, ihp.NMCR,
                                                 ihp.ssn, 0);
                std::tie(kxl, kyl, kzl) = EvalK0(false, ihp.Polarity, z, ihp.SolarPhase, ihp.SmoothTilt, ihp.NMCR,
                                                 ihp.ssn, 0);
            }
            hpps[j].push_back({
                kxh, kxl, kyh, kyl, kzh, kzl
            });
        }
        hps.push_back({
            ihp.V0 / aukm,
            g_low(ihp.SolarPhase, ihp.Polarity, ihp.SmoothTilt),
            rconst(ihp.SolarPhase, ihp.Polarity, ihp.SmoothTilt),
            ihp.TiltAngle * Pi / 180.f,
            static_cast<float>(ihp.Polarity) * sq(aum) * ihp.BEarth * 1e-9f / sqrtf(
                1.f + Omega * (1 - rhelio) / (ihp.V0 / aukm) * (Omega * (1 - rhelio) / (ihp.V0 / aukm))),
            EvalP0DriftSuppressionFactor(0, ihp.SolarPhase, ihp.TiltAngle, 0),
            EvalP0DriftSuppressionFactor(1, ihp.SolarPhase, ihp.TiltAngle, ihp.ssn),
            EvalHighRigidityDriftSuppression_plateau(ihp.SolarPhase, ihp.TiltAngle),
        });

        if (i < n_sources) {
            float mean_tilt = 0;
            for (size_t j = i; j < n_regions + i; ++j) mean_tilt += node_to_value<float>(static_node["tilt_angle"][j]);
            mean_tilt /= static_cast<float>(n_regions);

            high_activity.push_back(mean_tilt >= TiltL_MaxActivity_threshold);
            boundary.push_back({ihp.Rts_nose, ihp.Rhp_nose, ihp.Rts_tail, ihp.Rhp_tail});
        }
    }
    return {hpps, hps, high_activity, boundary};
}

/**
 * @brief Extract the heliosheat parameters from a YAML node
 * @param node the YAML node
 * @param size the number of sources
 * @return the heliosheat parameters vector
 */
vector<HeliosheatProperties_t> node_to_heliosheat(const fkyaml::node &node, const unsigned size) {
    const auto &static_node = node["static"]["heliosheat"];
    vector<HeliosheatProperties_t> hps;
    for (unsigned i = 0; i < size; ++i) {
        hps.push_back({
            node_to_value<float>(static_node["v0"][i]) / aukm,
            node_to_value<float>(static_node["k0"][i]),
        });
    }
    return hps;
}

/**
 * @brief Load the configuration file in YAML format
 * @param argc the number of arguments
 * @param argv the arguments
 * @param config the simulation configuration
 * @param verbose the verbosity level
 * @return EXIT_SUCCESS if the configuration was loaded successfully, EXIT_FAILURE otherwise
 */
int LoadConfigYaml(int argc, char *argv[], SimConfiguration_t &config, int verbose) {
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
            for (int i = 0; i < argc; ++i) { fprintf(stderr, "-->  %s \n", argv[i]); }
        }
    }

    auto node = fkyaml::node::deserialize(file);

    auto random_seed = node_to_value<unsigned long>(node["random_seed"]);
    auto output_path = node_to_value<std::string>(node["output_path"]);
    auto rigidities = node_to_vector<float>(node["rigidities"]);
    auto n_particles = node_to_value<unsigned>(node["n_particles"]);
    auto n_regions = node_to_value<unsigned>(node["n_regions"]);
    auto [isotopes_names, isotopes] = node_to_map<PartDescription_t>(node["isotopes"]);
    auto source_r = node_to_vector<float>(node["sources"]["r"]);
    auto source_th = node_to_vector<float>(node["sources"]["th"]);
    auto source_phi = node_to_vector<float>(node["sources"]["phi"]);
    auto n_sources = check_sources_count(source_r, source_th, source_phi);
    auto n_params = check_params_count(node["dynamic"]);
    auto isotopes_charge = check_isotopes_charge(isotopes);
    auto [heliosphere_param, heliosphere, high_activity, boundary] = node_to_heliosphere(
        node, n_sources, n_regions, n_params, isotopes_charge);
    auto heliosheat = node_to_heliosheat(node, n_sources);
    auto relative_bin_amplitude = node_to_value<float>(node["relative_bin_amplitude"]);

    config.output_file_name = output_path;
    config.RandomSeed = random_seed;
    config.Npart = n_particles;
    std::ranges::copy(rigidities, config.Tcentr = new float[config.NT = rigidities.size()]);

    config.NInitialPositions = n_sources;
    std::ranges::copy(source_r, config.InitialPositions.r = new float[n_sources]);
    std::ranges::copy(source_th, config.InitialPositions.th = new float[n_sources]);
    std::ranges::copy(source_phi, config.InitialPositions.phi = new float[n_sources]);

    //TODO: initialize correctly
    // config.Results = new MonteCarloResult_t[config.NT];
    config.RelativeBinAmplitude = relative_bin_amplitude;

    config.isotopes_names = isotopes_names;
    config.simulation_constants.NIsotopes = isotopes.size();
    std::ranges::copy(isotopes, config.simulation_constants.Isotopes);
    config.simulation_constants.Nregions = n_regions;
    std::ranges::copy(high_activity, config.simulation_constants.IsHighActivityPeriod);
    std::ranges::copy(boundary, config.simulation_constants.RadBoundary_real);

    std::ranges::copy(config.simulation_constants.RadBoundary_real, config.simulation_constants.RadBoundary_effe);
    for (unsigned i = 0; i < config.NInitialPositions; ++i) {
        RescaleToEffectiveHeliosphere(config.simulation_constants.RadBoundary_effe[i],
                                      config.InitialPositions, i);
    }

    config.simulation_parametrization.Nparams = n_params;
    config.simulation_parametrization.params = AllocateManaged<SimulationParametrizations_t::Parametrization_t
        []>(n_params);
    for (unsigned i = 0; i < n_params; ++i) {
        std::ranges::copy(heliosphere_param[i], config.simulation_parametrization.params[i].heliosphere);
    }
    std::ranges::copy(heliosphere, config.simulation_constants.heliosphere_properties);
    std::ranges::copy(heliosheat, config.simulation_constants.heliosheat_properties);

    if (verbose >= VERBOSE_med) {
        //TODO: change to VERBOSE_mid
        spdlog::trace("Nucleon Rest Mass : {:.3f}", config.simulation_constants.Isotopes[0].T0);
        spdlog::trace("Mass Number       : {:.1f}", config.simulation_constants.Isotopes[0].A);
        spdlog::trace("Charge            : {:.1f}", config.simulation_constants.Isotopes[0].Z);
        spdlog::trace("Number of sources : {}", static_cast<int>(config.NInitialPositions));


        spdlog::trace("Output file name: {}", config.output_file_name);
        spdlog::trace("Number of input energies: {}", config.NT);

        std::string energies_concat;
        for (const auto &T: std::span(config.Tcentr, config.NT)) {
            energies_concat += fmt::format("{:.2f} ", T);
        }
        spdlog::trace("Input energies: {}", energies_concat);


        spdlog::trace("Events to be generated: {}", config.Npart);
        spdlog::trace("For each simulated period:");

        for (unsigned i = 0; i < config.NInitialPositions; ++i) {
            spdlog::trace(
                "Position {:02}: (r: {:.2f}, theta: {:.2f}, phi: {:.2f}) {{{}, Rts: ({:.2f}, {:.2f}), Rhp: ({:.2f}, {:.2f})}}",
                i,
                config.InitialPositions.r[i], config.InitialPositions.th[i],
                config.InitialPositions.phi[i],
                config.simulation_constants.IsHighActivityPeriod[i] ? "High Activity" : "Low Activity",
                config.simulation_constants.RadBoundary_real[i].Rts_nose,
                config.simulation_constants.RadBoundary_real[i].Rts_tail,
                config.simulation_constants.RadBoundary_real[i].Rhp_nose,
                config.simulation_constants.RadBoundary_real[i].Rhp_tail);
        }


        spdlog::trace("Heliosphere Parameters ({} regions, {} sources, {} total):",
                      config.simulation_constants.Nregions, config.NInitialPositions,
                      config.simulation_constants.Nregions + config.NInitialPositions - 1);

        spdlog::trace("#Region: {{V0, k0_paral[], k0_perp[], GaussVar[], g_low, rconst, TiltAngle, Asun, P0d, P0dNS}}");

        for (unsigned i = 0; i < config.simulation_constants.Nregions + config.NInitialPositions - 1; ++i) {
            spdlog::trace(
                "{:02}: {{{:.3e}, [{:.3e}, {:.3e}], [{:.3e}, {:.3e}], [{:.3e}, {:.3e}], {:.3e}, {:.3e}, {:.3e}, {:.3e}, {:.3e}, {:.3e}}}",
                i,
                config.simulation_constants.heliosphere_properties[i].V0,
                config.simulation_parametrization.params[0].heliosphere[i].k0_paral[0],
                config.simulation_parametrization.params[0].heliosphere[i].k0_paral[1],
                config.simulation_parametrization.params[0].heliosphere[i].k0_perp[0],
                config.simulation_parametrization.params[0].heliosphere[i].k0_perp[1],
                config.simulation_parametrization.params[0].heliosphere[i].GaussVar[0],
                config.simulation_parametrization.params[0].heliosphere[i].GaussVar[1],
                config.simulation_constants.heliosphere_properties[i].g_low,
                config.simulation_constants.heliosphere_properties[i].rconst,
                config.simulation_constants.heliosphere_properties[i].TiltAngle,
                config.simulation_constants.heliosphere_properties[i].Asun,
                config.simulation_constants.heliosphere_properties[i].P0d,
                config.simulation_constants.heliosphere_properties[i].P0dNS);
        }

        spdlog::trace("Heliosheat Parameters ({} periods):", config.NInitialPositions);
        spdlog::trace("Period: {{V0, k0}}");
        for (unsigned ipos = 0; ipos < config.NInitialPositions; ++ipos) {
            spdlog::trace(
                "{:02}: {{{:.6e}, {:.6e}}}",
                ipos,
                config.simulation_constants.heliosheat_properties[ipos].V0,
                config.simulation_constants.heliosheat_properties[ipos].k0
            );
        }
    }
    return EXIT_SUCCESS;
}

/**
 * @brief Writes simulation results to a YAML file using fkYAML.
 *
 * This function builds a YAML document representing the simulation results stored in a 2D array
 * of MonteCarloResult_t structures.
 * @param filename   The name of the YAML output file to create or overwrite.
 * @param config    The simulation configuration containing the results to write.
 *
 * @return 0 on success, -1 on failure.
 */
int write_results_yaml(const std::string &filename, const SimConfiguration_t &config) {
    const unsigned NIsotopes = config.simulation_constants.NIsotopes;
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: cannot open file " << filename << " for writing." << std::endl;
        return -1;
    }

    auto root = fkyaml::node::mapping();
    auto &hist_seq = (root["histograms"] = fkyaml::node::sequence()).as_seq();

    for (unsigned i_rig = 0; i_rig < config.NT; ++i_rig) {
        auto &rig_entry = hist_seq.emplace_back(fkyaml::node{{"rigidity", config.Tcentr[i_rig]}});
        auto &isotopes = rig_entry["isotopes"] = fkyaml::node::mapping();
        for (unsigned i_iso = 0; i_iso < NIsotopes; ++i_iso) {
            auto &params = (isotopes[config.isotopes_names[i_iso]] = fkyaml::node::sequence()).as_seq();
            for (unsigned i_param = 0; i_param < config.simulation_parametrization.Nparams; ++i_param) {
                const auto &[n_reg, n_bins, low_b, ampl, bins] = config.Results[i_rig][i_param * NIsotopes + i_iso];
                params.emplace_back(fkyaml::node{
                    {"n_part", static_cast<int>(config.NInitialPositions * config.Npart)},
                    {"n_reg", static_cast<int>(n_reg)},
                    {"lower_bound", low_b},
                    {"amplitude", ampl},
                    {"bins", std::span(bins, n_bins)}
                });
            }
        }
    }
    ofs << root;

    return 0;
}
