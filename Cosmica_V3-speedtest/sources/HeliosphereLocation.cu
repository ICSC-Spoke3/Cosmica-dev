#include <math.h>
#include "SDECoeffs.cuh"
#include "VariableStructure.cuh"

__device__ int Zone(float r, float th, float phi){
    int zones[15] = {0};
    zones[14] = -1;

    int i = (int)r;

    return zones[i];
}