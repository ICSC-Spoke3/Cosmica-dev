#ifndef LoadConfigiguration
#define LoadConfigiguration

//Allocate the memory needed for an empty QuasiPArticle struct of N particle and return it
struct QuasiParticle_t InitQuasiPart_mem(int, int, bool);

// Load the initial position and rapidity of the quasi particle from configuration file
struct InitialPositions_t LoadInitPos(int, bool);

// Load the array of energy bins for which simulate the modulation
float *LoadInitRigidities(int, bool);

// Save propagation initial and final particle parameters to txt file
void SaveTxt_part(const char *, int, const struct QuasiParticle_t &, float, bool);

// Save the rigidities final histogram
void SaveTxt_histo(const char *, int, const struct MonteCarloResult_t &, bool);

#endif
