#include <math.h>
#include "SDECoeffs.cuh"
#include "VariableStructure.cuh"
#include "HelModVariableStructure.cuh"
#include "HeliosphereModel.cuh"

__device__ int Zone(int InitZone, float r, float th, float phi) {

    HeliosphereBoundRadius_t rbound = Heliosphere.RadBoundary_effe[InitZone];
	if (r<Boundary(th, phi,rbound.Rhp_nose, rbound.Rhp_tail)){
        // inside Heliopause boundary
        // inside Termination Shock Boundary
        return 1;
	}
    else {
        // outside heiosphere - Kill It
		return -1;
	}          
}