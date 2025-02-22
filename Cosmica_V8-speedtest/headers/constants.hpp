#pragma once

#define Pi      3.141592653589793f      // Pi
#define Half_Pi 1.570796326794896f      // Pi/2
#define aukm    149597870.691f          // 1 AU in km precise is 149.597.870,691 Km = 1e8 Km
#define aum     149597870691.f          // 1 AU in m precise is 149.597.870.691  m  = 1e11 m
#define aucm    14959787069100.f        // 1 AU in m precise is 149.597.870.691.00  cm  = 1e13 m
#define MeV     1e6f                    // MeV->eV                               eV
#define GeV     1e9f                    // MeV->eV                               eV
#define SoL       3e8f                    // Light Velodity                        m/s
#define thetaNorthlimit 0.000010f       // maximun value of latitude at north pole (this to esclude the region of not physical results due to divergence of equations)


#define thetaSouthlimit Pi-thetaNorthlimit // maximun value of latitude at south pole (this to esclude the region of not physical results due to divergence of equations)


#define struct_string_lengh 70
#define MaxCharinFileName   90
#define ReadingStringLenght 2000        // max lenght of each row while reading input file
//emulate bool
#define True    1
#define False   0

#define VERBOSE_low  1
#define VERBOSE_med  2
#define VERBOSE_hig  3

#ifndef PolarZone
#define PolarZone 30
#endif
#define CosPolarZone cosf(PolarZone*Pi/180.f)
#ifndef delta_m
#define delta_m 2.000000e-05f       // megnetic field disturbance in high latitude region
#endif
#ifndef TiltL_MaxActivity_threshold
#define TiltL_MaxActivity_threshold 50
#endif

// Solar param constant
#define Omega  3.03008e-6f  // solar angular velocity
#define rhelio 0.004633333f  // solar radius in AU
#define r_mirror 0.3