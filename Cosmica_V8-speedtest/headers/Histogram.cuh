#ifndef Histogram
#define Histogram

/*
// Execute the last steps of maximum search with unrolled reduction inside the synchronus warp
__device__ void WarpMax(volatile float *, unsigned);

// Execute the last steps of histogram reduction with unrolled reduction inside the synchronus warp
__device__ void WarpSum(volatile int *, unsigned);

// Execute maximum search with reduction algorithm inside the local block (with shared memory already allocated)
__device__ void BlockMax(float *, float *);

// Execute maximum search with reduction algorithm between the blocks
__global__ void GridMax(float, const float *, float *);

// Fill the rigidity block partial histogram with atomic sum
__global__ void Rhistogram_atomic(const float *, float, float, int, unsigned, float *);

// Merge the partial rigidity histograms with reduction algorithm (each block processes one rigidity bin)
__global__ void TotalHisto(float *, unsigned, unsigned, float *);

// Atomic partial histogram building for each block
__global__ void histogram_atomic(const float *, float, float, int, unsigned long, float *,
                                 int *);

// Total histogram bulding merging the partial block histograms
__global__ void histogram_accum(const float *, int, int, float *);
*/

__global__ void SimpleHistogram(ThreadIndexes_t, const float *, InstanceHistograms, unsigned *);

#endif
