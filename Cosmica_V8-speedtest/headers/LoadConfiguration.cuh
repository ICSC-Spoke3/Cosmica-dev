#ifndef LoadConfigiguration
#define LoadConfigiguration

template<typename T> requires (!std::is_array_v<T>)
T *AllocateManaged();

template<typename T> requires std::is_array_v<T>
std::remove_extent_t<T> *AllocateManaged(size_t);

template<typename T> requires (!std::is_array_v<T>)
auto AllocateManagedSafe();

template<typename T> requires std::is_array_v<T>
auto AllocateManagedSafe(size_t);

template<typename T> requires (!std::is_array_v<T>)
T *AllocateManaged(int);

template<typename T> requires std::is_array_v<T>
std::remove_extent_t<T> *AllocateManaged(size_t, int);

template<typename T> requires (!std::is_array_v<T>)
auto AllocateManagedSafe(int);

template<typename T> requires std::is_array_v<T>
auto AllocateManagedSafe(size_t, int);

ThreadQuasiParticles_t AllocateQuasiParticles(int);

InstanceHistograms *AllocateResults(unsigned, unsigned);

template<typename T>
void CopyToConstant(const T &, const T *);

ThreadIndexes_t AllocateIndex(int);

InitialPositions_t LoadInitPos(unsigned);

float *LoadInitRigidities(int, bool);

void SaveTxt_part(const char *, int, const ThreadQuasiParticles_t &, float);

void SaveTxt_histo(const char *, int, const MonteCarloResult_t &);

#endif
