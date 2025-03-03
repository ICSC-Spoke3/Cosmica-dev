#ifndef GPUManage
#define GPUManage

void HandleError(cudaError_t, const char *, int);

constexpr unsigned long hash(const std::string_view &);

consteval unsigned long operator""_(const char *, const size_t);

int AvailableGPUs();

int BestNWarpPerBlock(char, bool);

LaunchParam_t RoundNpart(unsigned, cudaDeviceProp, bool, int);

cudaDeviceProp *DeviceInfo(int);


#endif
