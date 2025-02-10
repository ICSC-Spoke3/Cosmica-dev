__device__ int Zone(const float r, float th, float phi) {
    int zones[15] = {0};
    zones[14] = -1;

    const int i = static_cast<int>(r);

    return zones[i];
}
