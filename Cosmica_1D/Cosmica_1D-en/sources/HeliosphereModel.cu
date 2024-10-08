#include "GenComputation.cuh"
#include "HeliosphereModel.cuh"
#include "HelModVariableStructure.cuh"
#include "VariableStructure.cuh"
//  Nose in Heliocentric Inertial (HCI o HGI) cartesian  
#define nose_x -0.996
#define nose_y 0.03
#define nose_z 0.088
__host__ __device__ float Boundary(float theta, float phi, float a, float b)
{ /* Author: SDT - Mar 2018 -- updated Apr 2018 (formula from GLV) -- adapted for CUDA in Feb 2022
   * description: check if position is outside the spheroid that define the Heliosphere boundary. 
   *              The spheroid is defined as two different spheres, one of radius b (tail distance) the second of radius a (nose distance). 
   *              The transition between the two spheres is defined with the cosine^2 of the angle between the direction and the heliosphere Nose
   * Nose in ecliptic coord (degree) lat,lon  [ 5.3 254.7 ]
   * Nose in Heliocentric Inertial (HCI o HGI) cartesian  [ -0.996 0.03  0.088] HCI lat, long  [   5.121  178.269]
   */
  float x=cos(phi)*sin(theta);
  float y=sin(phi)*sin(theta);
  float z=cos(theta);
  float cosAlpha=x*nose_x+y*nose_y+z*nose_z;
  if (cosAlpha>0) 
  {
  	return b-(b-a)*cosAlpha*cosAlpha;
  }else{
    return b;
  }
}


__device__ float RadialZone(int InitZone, float r, float th, float phi){
/* Author: SDT - Feb 2022
   * description: find in which zone the particle is.
   *              This function assume that, inside the Termination Shock, the heliosphere is divided in 
   *              equally spatial Heliosphere.Nregions regions. outside TS there is 1 region.
   *              Each region is circular shaped, this means that last region is broader in tail direction.
   * return :  Number of zone between 0 and Heliosphere.Nregions-1 if inside TS
               Heliosphere.Nregions if inside Heliosheat
               -1 if outside Heliopause
   * // NOTE DEV rispetto alla versione HelMod si uniscono le funzioni RadialZone e WhereIAm
   */
	
	HeliosphereBoundRadius_t rbound = Heliosphere.RadBoundary_effe[InitZone];
	if (r<Boundary(th, phi,rbound.Rhp_nose, rbound.Rhp_tail))
	{   // inside Heliopause boundary
		if (r>=Boundary(th, phi,rbound.Rts_nose, rbound.Rts_tail))
		{   // inside Heliosheat
			return Heliosphere.Nregions;
		}else{
			// inside Termination Shock Boundary
			if (r<rbound.Rts_nose) return floor(r/rbound.Rts_nose*Heliosphere.Nregions);
			else                             return Heliosphere.Nregions-1;  
		}
	}else{ // outside heiosphere - Kill It
		return -1;
	}                
}
