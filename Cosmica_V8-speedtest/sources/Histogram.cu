// .. credit to Mark Harris (https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

// Use of the tamplate to unroll the loop at compile time (kernel_6 optimization)
/* template <unsigned int blockSize>
__device__ void WarpMax(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64 && sdata[tid] < sdata[tid + 32]) sdata[tid] = sdata[tid + 32];
    if (blockSize >= 32 && sdata[tid] < sdata[tid + 16]) sdata[tid] = sdata[tid + 16];
    if (blockSize >= 16 && sdata[tid] < sdata[tid + 8])  sdata[tid] = sdata[tid + 8];
    if (blockSize >= 8  && sdata[tid] < sdata[tid + 4])  sdata[tid] = sdata[tid + 4];
    if (blockSize >= 4  && sdata[tid] < sdata[tid + 2])  sdata[tid] = sdata[tid + 2];
    if (blockSize >= 2  && sdata[tid] < sdata[tid + 1])  sdata[tid] = sdata[tid + 1];
}

template <unsigned int blockSize>
__device__ void BlockMax(float* sdata, float* outdata) {
    
    // thread index taking into account the shift imposed by the rigidity positions in shared memory array
    unsigned int tid = threadIdx.x + 3*blockSize;

    // first max search steps with sub-array larger than warp dimension (unrolled loop of max search)
    if (blockSize >= 512) {
        if (tid < 256 && sdata[tid] < sdata[tid + 256]) sdata[tid] = sdata[tid + 256];
            __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128 && sdata[tid] < sdata[tid + 128]) sdata[tid] = sdata[tid + 128];
            __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64 && sdata[tid] < sdata[tid + 64]) sdata[tid] = sdata[tid + 64];
            __syncthreads();
    }

    // warp reduction
    if (tid < 32) WarpMax(sdata, tid);

    // write result for this block to global mem
    if (tid == 0) outdata = sdata[0];
}

template <unsigned int blockSize>
__global__ void GridMax(float* indata, float* outdata) {
    // shared memory allocation and filling
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize + threadIdx.x;

    sdata[tid] = indata[i];
    __syncthreads();

    // first max search steps with sub-array larger than warp dimension (unrolled loop of max search)
    if (blockSize >= 512) {
        if (tid < 256 && sdata[tid] < sdata[tid + 256]) sdata[tid] = sdata[tid + 256];
            __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128 && sdata[tid] < sdata[tid + 128]) sdata[tid] = sdata[tid + 128];
            __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64 && sdata[tid] < sdata[tid + 64]) sdata[tid] = sdata[tid + 64];
            __syncthreads();
    }

    // warp reduction
    if (tid < 32) WarpMax(sdata, tid);

    // write result for this block to global mem
    if (tid == 0) outdata = sdata[0];
} */

// Unroll the last steps when reduction dimension < warp dimension
__device__ void WarpMax(volatile float *sdata, const unsigned int tid) {
    if (sdata[tid] < sdata[tid + 32]) sdata[tid] = sdata[tid + 32];
    if (sdata[tid] < sdata[tid + 16]) sdata[tid] = sdata[tid + 16];
    if (sdata[tid] < sdata[tid + 8]) sdata[tid] = sdata[tid + 8];
    if (sdata[tid] < sdata[tid + 4]) sdata[tid] = sdata[tid + 4];
    if (sdata[tid] < sdata[tid + 2]) sdata[tid] = sdata[tid + 2];
    if (sdata[tid] < sdata[tid + 1]) sdata[tid] = sdata[tid + 1];
}

__device__ void BlockMax(float *sdata, float *outdata) {
    // thread index taking into account the shift imposed by the rigidity positions in shared memory array
    const unsigned int sdata_id = threadIdx.x + 3 * blockDim.x;

    // first max search steps with sub-array larger than warp dimension
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s && sdata[sdata_id] < sdata[sdata_id + s]) sdata[sdata_id] = sdata[sdata_id + s];
        __syncthreads();
    }

    // warp reduction
    if (threadIdx.x < 32) WarpMax(sdata, sdata_id);

    // write result for this block to global mem
    if (threadIdx.x == 0) outdata[blockIdx.x] = sdata[sdata_id];
}

__global__ void GridMax(const int Nmax, const float *indata, float *outdata) {
    // shared memory allocation and filling
    extern __shared__ float sdata[];

    // !!!This is useful only launched recursively on different blocks
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < Nmax) sdata[threadIdx.x] = indata[id];
    else if (id >= Nmax) sdata[threadIdx.x] = 0;

    __syncthreads();

    // first max search steps with sub-array larger than warp dimension
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s && sdata[threadIdx.x] < sdata[threadIdx.x + s]) sdata[threadIdx.x] = sdata[threadIdx.x + s];
        __syncthreads();
    }

    // warp reduction
    if (threadIdx.x < 32) WarpMax(sdata, threadIdx.x);

    // write result for this block to global mem
    if (threadIdx.x == 0) outdata[blockIdx.x] = sdata[0];
}

__global__ void Rhistogram_atomic(const float *R_in, const float LogBin0_lowEdge, const float DeltaLogR, const int Nbin,
                                  const unsigned int Npart, float *R_out) {
    extern __shared__ unsigned int smem[];

    const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int block_shift = blockIdx.x * Nbin;

    // initialize the shared memory empty histogram
    if (threadIdx.x < Nbin) smem[threadIdx.x] = 0;

    __syncthreads();

    if (id < Npart) {
        if (log10f(R_in[id]) > LogBin0_lowEdge) {
            // evalaute the bin where put event and add atomically
            const int dest_bin = static_cast<int>(floorf((log10f(R_in[id]) - LogBin0_lowEdge) / DeltaLogR));
            atomicAdd(&smem[dest_bin], 1);
        }
    }

    // write partial histogram to global memory
    if (threadIdx.x < Nbin) R_out[threadIdx.x + block_shift] = static_cast<float>(smem[threadIdx.x]);
}

// Unroll the last steps when reduction dimension < warp dimension
__device__ void WarpSum(volatile int *sdata, const unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void TotalHisto(const float *indata, const unsigned int Nbins, const unsigned int Nblocks, float *outdata) {
    // shared memory allocation and filling
    extern __shared__ int shist[];

    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int bin_id = 2 * threadIdx.x * Nbins + blockIdx.x;

    // First histogram couple merge during shared memory allocation
    // Each block perform one rigidity bin reduction
    if (id < Nbins * Nblocks) shist[threadIdx.x] = static_cast<int>(indata[bin_id] + indata[bin_id + Nbins]);

    __syncthreads();

    // first max search steps with sub-array larger than warp dimension
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) shist[threadIdx.x] += shist[threadIdx.x + s];
        __syncthreads();
    }

    // warp reduction
    if (threadIdx.x < 32) WarpSum(shist, threadIdx.x);

    // write result for this block to global mem
    if (threadIdx.x == 0) outdata[blockIdx.x] = static_cast<float>(shist[0]);
}

////////////////////////////////////////////////////////////////
//..... histogram handling .....................................
//
////////////////////////////////////////////////////////////////

__global__ void histogram_atomic(const float *in, const float LogBin0_lowEdge, const float DeltaLogT, const int Nbin,
                                 const unsigned long Npart, float *out, int *Nfailed) {
    // NOTE: not using shared memory

    const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    for (unsigned int it = threadIdx.x; it < Nbin; it += blockDim.x) {
        out[blockIdx.x * Nbin + it] = 0;
    }

    __syncthreads();

    if (id < Npart) {
        if (log10f(in[id]) > LogBin0_lowEdge) {
            const int DestBin = static_cast<int>(floorf((log10f(in[id]) - LogBin0_lowEdge) / DeltaLogT));
            // evalaute the bin where put event
            atomicAdd(&out[blockIdx.x * Nbin + DestBin], 1); // exp(alphapath[id])
        } else {
            atomicAdd(Nfailed, 1);
            // nota per futuro. le particelle uccise hanno valori diversi negativi in base all'evento che li ha uccisi,
            //                  quindi se Nfailed diventasse una struct con il tipo di errore, si potrebbe fare una
            //                  statistica dettagliata dell'errore.
        }
    }
}

__global__ void histogram_accum(const float *in, const int Nbins, const int NFraz, float *out) {
    // NOTE: not using shared memory

    const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= Nbins) { return; } //out of range

    float total = 0.;

    for (int ithb = 0; ithb < NFraz; ithb++) {
        total += in[id + ithb * Nbins];
    }

    out[id] = total;
}
