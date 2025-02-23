#ifndef LoadConfigiguration
#define LoadConfigiguration

template<typename T>
T *AllocateManaged(size_t);

template<typename T> requires (!std::is_array_v<T>)
auto AllocateManagedSafe();

// Overload for array types.
template<typename T> requires std::is_array_v<T>
auto AllocateManagedSafe(size_t);

template<typename T>
T *AllocateManaged(size_t, int);

template<typename T> requires (!std::is_array_v<T>)
auto AllocateManagedSafe(int v);

// Overload for array types.
template<typename T> requires std::is_array_v<T>
auto AllocateManagedSafe(size_t, int);

//Allocate the memory needed for an empty QuasiPArticle struct of N particle and return it
ThreadQuasiParticles_t AllocateQuasiParticles(int);

template<typename T>
void CopyToConstant(const T &, const T *);

ThreadIndexes_t AllocateIndex(int);

// Load the initial position and rapidity of the quasi particle from configuration file
InitialPositions_t LoadInitPos(unsigned, bool);

// Load the array of energy bins for which simulate the modulation
float *LoadInitRigidities(int, bool);

// Save propagation initial and final particle parameters to txt file
void SaveTxt_part(const char *, int, const struct ThreadQuasiParticles_t &, float, bool);

// Save the rigidities final histogram
void SaveTxt_histo(const char *, int, const struct MonteCarloResult_t &, bool);

#endif
