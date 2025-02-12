#include <cstdio>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions
#include "LoadConfiguration.cuh"
#include "VariableStructure.cuh"

QuasiParticle_t InitQuasiPart_mem(const int Npart, const int hardware, const bool verbose) {
    // initialize the QuasiPArticle struct fields
    QuasiParticle_t empty_QuasiPart;

    if (hardware == 0) {
        // Allocate the needed memory for the variables arrays with custom dimension on the host
        empty_QuasiPart.r = new float[Npart];
        empty_QuasiPart.th = new float[Npart];
        empty_QuasiPart.phi = new float[Npart];
        empty_QuasiPart.R = new float[Npart];
        empty_QuasiPart.t_fly = new float[Npart];
        // empty_QuasiPart.alphapath = (float*)malloc(Npart*sizeof(float));

        if (verbose) {
            printf("Corrected initialized quasi particle empty array in the host\n");
        }
    } else if (hardware == 1) {
        // Allocate the needed memory for the variables arrays with custom dimension on the device
        cudaMalloc(reinterpret_cast<void **>(&empty_QuasiPart.r), Npart * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&empty_QuasiPart.th), Npart * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&empty_QuasiPart.phi), Npart * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&empty_QuasiPart.R), Npart * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&empty_QuasiPart.t_fly), Npart * sizeof(float));
        // cudaMalloc((void**)&empty_QuasiPart.alphapath, Npart*sizeof(float));

        if (verbose) {
            printf("Corrected initialized quasi particle empty array in the device\n");
        }
    } else {
        printf("Insert hardware variable to choose if the memeory must be allocated in the host or device\n");
    }


    return empty_QuasiPart;
}

InitialPositions_t LoadInitPos(int Npos, const bool verbose) {
    // Allocate the array memory
    InitialPositions_t InitialPositions;

    InitialPositions.r = new float[Npos];
    InitialPositions.th = new float[Npos];
    InitialPositions.phi = new float[Npos];

    // Set the InitialPositions arrays with default values

    if (verbose) {
        printf("Default initial quasi particles configuration loaded\n");
    }

    return InitialPositions;
}

float *LoadInitRigidities(const int RBins, const bool verbose) {
    // Allocate the array memory
    auto *rigidities = new float[RBins];

    // Set the InitialPositions arrays with default values
    for (int i = 0; i < RBins; i++) {
        rigidities[i] = sqrtf(static_cast<float>(i) + 1);
    }

    if (verbose) {
        printf("Default energies bins loaded\n");
    }

    return rigidities;
}

void SaveTxt_part(const char *filename, const int Npart, const QuasiParticle_t &Out_QuasiParts, const float RMax,
                  const bool verbose) {
    FILE *file = fopen(filename, "ab");
    if (file == nullptr) {
        printf("Error opening the file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fprintf(file, "r\t\t\t theta\t\t phi\t\t rigidity\t fly time\n"); // \t\t alphapath

    for (int i = 0; i < Npart; i++) {
        fprintf(file, "%f\t %f\t %f\t %f\t %f\n", Out_QuasiParts.r[i], Out_QuasiParts.th[i],
                Out_QuasiParts.phi[i], Out_QuasiParts.R[i], Out_QuasiParts.t_fly[i]); // Out_QuasiParts.alphapath[i]
    }

    fprintf(file, "max R\n%f\n\n", RMax);

    if (verbose) {
        printf("Propagation variables written successfully in file %s\n", filename);
    }

    fclose(file);
}

void SaveTxt_histo(const char *filename, const int Bins, const MonteCarloResult_t &histo, const bool verbose) {
    FILE *file = fopen(filename, "ab");
    if (file == nullptr) {
        printf("Error opening the file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fprintf(file, "Nbins = %d\t LogBin0_lowEdge = %f\t LogBin_Rmax = %f\t DeltaLogR = %f\n", histo.Nbins,
            histo.LogBin0_lowEdge, static_cast<float>(histo.Nbins) * histo.DeltaLogR, histo.DeltaLogR);
    fprintf(file, "Histogram counts:\n");

    for (int i = 0; i < Bins; i++) {
        fprintf(file, "%f\n", histo.BoundaryDistribution[i]);
    }

    if (verbose) {
        printf("Histo array written successfully\n");
    }

    fclose(file);
}
