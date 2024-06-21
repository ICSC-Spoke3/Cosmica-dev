#include <stdio.h>          // Supplies FILE, stdin, stdout, stderr, and the fprint() family of functions
#include "LoadConfiguration.cuh"
#include "VariableStructure.cuh"

struct QuasiParticle_t InitQuasiPart_mem(int Npart, int hardware, bool verbose) {

    // initialize the QuasiPArticle struct fields
    struct QuasiParticle_t empty_QuasiPart;
    
    if (hardware==0) {

        // Allocate the needed memory for the variables arrays with custom dimension on the host
        empty_QuasiPart.r = (float*)malloc(Npart*sizeof(float));
        empty_QuasiPart.th = (float*)malloc(Npart*sizeof(float));
        empty_QuasiPart.phi = (float*)malloc(Npart*sizeof(float));
        empty_QuasiPart.R = (float*)malloc(Npart*sizeof(float));
        empty_QuasiPart.t_fly = (float*)malloc(Npart*sizeof(float));
        empty_QuasiPart.alphapath = (float*)malloc(Npart*sizeof(float));

        if (verbose) {
            printf("Corrected initialized quasi particle empty array in the host\n");
        }
    }

    else if (hardware==1) {

        // Allocate the needed memory for the variables arrays with custom dimension on the device
        cudaMalloc((void**)&empty_QuasiPart.r, Npart*sizeof(float));
        cudaMalloc((void**)&empty_QuasiPart.th, Npart*sizeof(float));
        cudaMalloc((void**)&empty_QuasiPart.phi, Npart*sizeof(float));
        cudaMalloc((void**)&empty_QuasiPart.R, Npart*sizeof(float));
        cudaMalloc((void**)&empty_QuasiPart.t_fly, Npart*sizeof(float));
        cudaMalloc((void**)&empty_QuasiPart.alphapath, Npart*sizeof(float));

        if (verbose) {
            printf("Corrected initialized quasi particle empty array in the device\n");
        }
    }

    else {
        printf("Insert hardware variable to choose if the memeory must be allocated in the host or device\n");
    }

    

    return empty_QuasiPart;
}

struct InitialPositions_t LoadInitPos(int Npos, bool verbose, bool trivial) {
    
    // Allocate the array memory
    struct InitialPositions_t InitialPositions;

    InitialPositions.r = (float*)malloc(Npos*sizeof(float));
    InitialPositions.th = (float*)malloc(Npos*sizeof(float));
    InitialPositions.phi = (float*)malloc(Npos*sizeof(float));

    // Set the InitialPositions arrays with default values
    if (trivial) {    
        for (int i=0; i<Npos; i++) {
            InitialPositions.r[i] = i;
            InitialPositions.th[i] = M_PI_2;
            InitialPositions.phi[i] = 0;
        }
    }

    if (verbose) {
        printf("Default initial quasi particles configuration loaded\n");
    }

    return InitialPositions;
}

float* LoadInitRigidities(int RBins, bool verbose) {

    // Allocate the array memory
    float* rigidities = (float*)malloc(RBins*sizeof(float));

    // Set the InitialPositions arrays with default values
    for (int i=0; i<RBins; i++) {
        rigidities[i] = (float)sqrt(i+1);
    }

    if (verbose) {
        printf("Default energies bins loaded\n");
    }

    return rigidities;
}

void SaveTxt_part(const char* filename, int Npart, QuasiParticle_t Out_QuasiParts, float RMax, bool verbose) {

    FILE* file = fopen(filename, "ab");
    if (file == NULL)
    {
        printf("Error opening the file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    else {
        fprintf(file, "r\t\t\t theta\t\t phi\t\t rigidity\t fly time\t\t alphapath\n");

        for (int i=0; i<Npart; i++) {
            fprintf(file, "%f\t %f\t %f\t %f\t %f\t\t %f\n", Out_QuasiParts.r[i], Out_QuasiParts.th[i], Out_QuasiParts.phi[i], Out_QuasiParts.R[i], Out_QuasiParts.t_fly[i], Out_QuasiParts.alphapath[i]);
        }

        fprintf(file, "max R\n%f\n\n", RMax);
            
        if (verbose) {
            printf("Propagation variables written successfully in file %s\n", filename);
        }
    }

    fclose(file);
}

void SaveTxt_histo(const char* filename, int Bins, MonteCarloResult_t histo, bool verbose) {

    FILE* file = fopen(filename, "ab");
    if (file == NULL)
    {
        printf("Error opening the file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    else {
        fprintf(file, "Nbins = %d\t LogBin0_lowEdge = %f\t LogBin_Rmax = %f\t DeltaLogR = %f\n", histo.Nbins, histo.LogBin0_lowEdge, histo.Nbins*histo.DeltaLogR, histo.DeltaLogR);
        fprintf(file, "Histogram counts:\n");

        for (int i=0; i<Bins; i++) {
            fprintf(file, "%f\n", histo.BoundaryDistribution[i]);
        }
            
        if (verbose) {
            printf("Histo array written successfully\n");
        }
    }

    fclose(file);
}