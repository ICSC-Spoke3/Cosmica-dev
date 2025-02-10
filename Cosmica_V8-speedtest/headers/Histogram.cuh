#ifndef Histogram
#define Histogram

// Execute the last steps of maximum search with unrolled reduction inside the synchronus warp
__device__ void WarpMax(volatile float *, unsigned int);

// Execute the last steps of histogram reduction with unrolled reduction inside the synchronus warp
__device__ void WarpSum(volatile int *, unsigned int);

// Execute maximum search with reduction algorithm inside the local block (with shared memory already allocated)
__device__ void BlockMax(float *, float *);

// Execute maximum search with reduction algorithm between the blocks
__global__ void GridMax(const float, const float *, float *);

// Fill the rigidity block partial histogram with atomic sum
__global__ void Rhistogram_atomic(const float *, const float, const float, const int, const unsigned int, float *);

// Merge the partial rigidity histograms with reduction algorithm (each block processes one rigidity bin)
__global__ void TotalHisto(float *, const unsigned int, const unsigned int, float *);

// Atomic partial histogram building for each block
__global__ void histogram_atomic(const float *, const float, const float, const int, const unsigned long, float *,
                                 int *);

// Total histogram bulding merging the partial block histograms
__global__ void histogram_accum(const float *, const int, const int, float *);

#include "Histogram.cuh"

#endif
