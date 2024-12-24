# -*- coding: utf-8 -*-
# developed: Dec 2022
# last update: Dec 2022
''' 
  description: Procedura che converte i risultati del montecarlo in un unico file npz

'''
# -- libraries
import numpy as np
import os                                 # OS directory manager
import glob
import pickle
import astropy.io.fits as pyfits # per elaborare i fits file
from scipy.interpolate import interp1d # interpolazione
# -- directory e nomi file generali
LISPATH_root="/DISK/eliosfera/WebSiteHelMod/LIS/"
LISPATH_dict={
    "Antiproton":"ApJ_ProtonHeAntip/LIS_Default_Antiproton.gz",
    "Proton":    "ApJ_28Nuclei/LIS_Default2020_Proton.gz",
    "H2":        "ApJ_28Nuclei/LIS_Default2020_Proton.gz",
    "H1":        "ApJ_28Nuclei/LIS_Default2020_Proton.gz",    
    "Helium":    "ApJ_28Nuclei/LIS_Default2020_Helium.gz",
    "He3":      "ApJ_28Nuclei/LIS_Default2020_Helium.gz",
    "He4":      "ApJ_28Nuclei/LIS_Default2020_Helium.gz",
    "Lithium":   "ApJ_28Nuclei/LIS_Default2020_Lithium.gz",
    "Beryllium": "ApJ_28Nuclei/LIS_Default2020_Beryllium.gz",
    "Boron":     "ApJ_28Nuclei/LIS_Default2020_Boron.gz",
    "Carbon":    "ApJ_28Nuclei/LIS_Default2020_Carbon.gz",
    "Nitrogen":  "ApJ_28Nuclei/LIS_Default2020_Nitrogen.gz",
    "Oxygen":    "ApJ_28Nuclei/LIS_Default2020_Oxygen.gz",
    "Fluorine":  "ApJ_Fluorine_Total/LIS_Default2021-total_Fluorine.gz", 
    "Neon":      "ApJ_28Nuclei/LIS_Default2020_Neon.gz",
    "Sodium":    "ApJ_NaAl_default/LIS_Default2022_Sodium.gz",
    "Magnesium": "ApJ_28Nuclei/LIS_Default2020_Magnesium.gz",
    "Aluminum":  "ApJ_NaAl_total/LIS_Default2022-total_Aluminum.gz",
    "Silicon":   "ApJ_28Nuclei/LIS_Default2020_Silicon.gz",
    "Phosphorus":"ApJ_28Nuclei/LIS_Default2020_Phosphorus.gz",
    "Sulfur":    "ApJ_28Nuclei/LIS_Default2020_Sulfur.gz",
    "Chlorine":  "ApJ_28Nuclei/LIS_Default2020_Chlorine.gz",
    "Argon":     "ApJ_28Nuclei/LIS_Default2020_Argon.gz",
    "Potassium": "ApJ_28Nuclei/LIS_Default2020_Potassium.gz",
    "Calcium":   "ApJ_28Nuclei/LIS_Default2020_Calcium.gz",
    "Scandium":  "ApJ_28Nuclei/LIS_Default2020_Scandium.gz",
    "Titanium":  "ApJ_28Nuclei/LIS_Default2020_Titanium.gz",
    "Vanadium":  "ApJ_28Nuclei/LIS_Default2020_Vanadium.gz",
    "Chromium":  "ApJ_28Nuclei/LIS_Default2020_Chromium.gz",
    "Manganese": "ApJ_28Nuclei/LIS_Default2020_Manganese.gz",
    "Iron":      "ApJ_Iron/LIS_Default2021_Iron.gz",
    "Cobalt":    "ApJ_28Nuclei/LIS_Default2020_Cobalt.gz",
    "Nickel":    "ApJ_28Nuclei/LIS_Default2020_Nickel.gz",    
}
#LISPATH="/DISK/eliosfera/Analysis/2022/HelMod4Sim/LISs/2022_artNaAL_PUBBLICATO_TOTALE_nuclei_57_NaF_Alsig5_NoSigs_SummedAl"
#LISPATH="/DISK/eliosfera/Analysis/2021/HelMod4Sim/LISs/2021_ArtFluo_nuclei_56_BEST_Li_4_BoC11_hnuc8_8_8_nucleon7_Fe8_3bis_NaF_new1_sig.gz"
EXELIST="exefiles/CodeVersions.txt"
SIMLIST="Simulations.list"
FORCE_RECALC=False # forza il ricalcolo dei file di output (if false -> skip Ions if RawMatrix File already exist)
HMColorLine='tab:red'
# ---- Isotopes List ------------
Isotopes_dict ={              #  Z    A   T0[GeV/n]      Name  
                "Electron":  [( -1.,  1., 5.109989e-04,"Electron"  )],
                "Antiproton":[( -1.,  1., 0.938272    ,"Antiproton")],
                "Positron":  [(  1.,  1., 5.109989e-04,"Positron"  )],
                "Proton":    [(  1.,  1., 0.938272    ,"Proton"    ),
                              (  1.,  2., 0.938272    ,"Deuteron"  )],
                "H1":        [(  1.,  1., 0.938272    ,"H1"        )],
                "H2":        [(  1.,  2., 0.938272    ,"H2"        )],
                "Helium":    [(  2.,  4., 0.931494061 ,"Helium"    ),
                              (  2.,  3., 0.931494061 ,"Heli3"     )],
                "He4":       [(  2.,  4., 0.931494061 ,"He4"       )],
                "He3":       [(  2.,  3., 0.931494061 ,"He3"       )],
                "Lithium":   [(  3.,  7., 0.931494061 ,"Lithium"   ),
                              (  3.,  6., 0.931494061 ,"Lith6"     )],
                "Li6":       [(  3.,  6., 0.931494061 ,"Li6"       )],
                "Li7":       [(  3.,  7., 0.931494061 ,"Li7"       )],
                "Beryllium": [(  4.,  9., 0.931494061 ,"Beryllium" ),
                              (  4.,  7., 0.931494061 ,"Beryl7"    ),
                              (  4., 10., 0.931494061 ,"Beryl10"   )],
                "Be10":      [(  4., 10., 0.931494061 ,"Be10"      )],
                "Be9":       [(  4.,  9., 0.931494061 ,"Be9"       )],
                "Be7":       [(  4.,  7., 0.931494061 ,"Be7"       )],
                "Boron":     [(  5., 11., 0.931494061 ,"Boron"     ),
                              (  5., 10., 0.931494061 ,"Bor10"     )],
                "B10":       [(  5., 10., 0.931494061 ,"B10"       )],
                "B11":       [(  5., 11., 0.931494061 ,"B11"       )],
                "Carbon":    [(  6., 12., 0.931494061 ,"Carbon"    ),
                              (  6., 13., 0.931494061 ,"Carb13"    )],
                              #(  6., 14., 0.931494061 ,"Carb14"    )],
                "C12":       [(  6., 12., 0.931494061 ,"C12"       )],
                "C13":       [(  6., 13., 0.931494061 ,"C13"       )],
                "C14":       [(  6., 14., 0.931494061 ,"C14"       )],
                "Nitrogen":  [(  7., 14., 0.931494061 ,"Nitrogen"  ),
                              (  7., 15., 0.931494061 ,"Nitro15"   )],
                "N14":       [(  7., 14., 0.931494061 ,"N14"       )],
                "N15":       [(  7., 15., 0.931494061 ,"N15"       )],                             
                "Oxygen":    [(  8., 16., 0.931494061 ,"Oxygen"    ),
                              (  8., 17., 0.931494061 ,"Oxyg17"    ),
                              (  8., 18., 0.931494061 ,"Oxyg18"    )],
                "O16":       [(  8., 16., 0.931494061 ,"O16"       )],
                "O17":       [(  8., 17., 0.931494061 ,"O17"       )],
                "O18":       [(  8., 18., 0.931494061 ,"O18"       )],       
                "Fluorine":  [(  9., 19., 0.931494061 ,"Fluorine"  )],
                "F19":       [(  9., 19., 0.931494061 ,"F19"       )],                                       
                "F18":       [(  9., 18., 0.931494061 ,"F18"       )],                                       

                "Neon":      [( 10., 20., 0.931494061 ,  "Neon"    ),
                              ( 10., 21., 0.931494061 ,  "Neo21"   ),
                              ( 10., 22., 0.931494061 ,  "Neo22"   )],
                "Ne20":      [( 10., 20., 0.931494061 ,  "Ne20"    )],
                "Ne21":      [( 10., 21., 0.931494061 ,  "Ne21"    )],
                "Ne22":      [( 10., 22., 0.931494061 ,  "Ne22"    )],
                "Sodium":    [( 11., 23., 0.931494061 ,"Sodium"    )],
                              #( 11., 22., 0.931494061 ,"Sodi22"    )],
                "Na23":      [( 11., 23., 0.931494061 ,"Na23"      )],
                "Magnesium": [( 12., 24., 0.931494061 ,"Magnesium" ),
                              ( 12., 25., 0.931494061 ,"Magn25"    ),
                              ( 12., 26., 0.931494061 ,"Magn26"    )],
                "Mg24":      [( 12., 24., 0.931494061 ,"Mg24"      )],
                "Mg25":      [( 12., 25., 0.931494061 ,"Mg25"      )],
                "Mg26":      [( 12., 26., 0.931494061 ,"Mg26"      )],
                "Aluminum":  [( 13., 27., 0.931494061 ,"Aluminum"  ),
                              ( 13., 26., 0.931494061 ,"Alum26"    )],
                "Al27":      [( 13., 27., 0.931494061 ,"Al27"      )],
                "Al26":      [( 13., 26., 0.931494061 ,"Al26"      )],    
                "Silicon":   [( 14., 28., 0.931494061 ,"Silicon"   ),
                              ( 14., 29., 0.931494061 ,"Silic29"   ),
                              ( 14., 30., 0.931494061 ,"Silic30"   )],
                "Si28":      [( 14., 28., 0.931494061 ,"Silicon"   )],
                "Si29":      [( 14., 29., 0.931494061 ,"Silic29"   )],
                "Si30":      [( 14., 30., 0.931494061 ,"Silic30"   )],
                "Phosphorus":[( 15., 31., 0.931494061 ,"Phosphorus")],
                              #( 15., 32., 0.931494061 ,"Phos32"    ),
                              #( 15., 33., 0.931494061 ,"Phos33"    )],
                "P31":       [( 15., 31., 0.931494061 ,"P31"       )],
                "P32":       [( 15., 32., 0.931494061 ,"P32"       )],
                "P33":       [( 15., 33., 0.931494061 ,"P33"       )],
                "Sulfur":    [( 16., 32., 0.931494061 ,"Sulfur"    ),
                              ( 16., 33., 0.931494061 ,"Sulf33"    ),
                              ( 16., 34., 0.931494061 ,"Sulf34"    ),
                              #( 16., 35., 0.931494061 ,"Sulf35"    ),
                              ( 16., 36., 0.931494061 ,"Sulf36"    )],
                "S32":       [( 15., 32., 0.931494061 ,"S32"       )],
                "S33":       [( 16., 33., 0.931494061 ,"S33"       )],
                "S34":       [( 16., 34., 0.931494061 ,"S34"       )],
                "S35":       [( 16., 35., 0.931494061 ,"S35"       )],
                "S36":       [( 16., 36., 0.931494061 ,"S36"       )],
                "Chlorine":  [( 17., 35., 0.931494061 ,"Chlorine"  ),
                              ( 17., 36., 0.931494061 ,"Chlo36"    ),
                              ( 17., 37., 0.931494061 ,"Chlo37"    )],
                "Cl35":      [( 17., 35., 0.931494061 ,"Cl35"      )],
                "Cl36":      [( 17., 36., 0.931494061 ,"Cl36"      )],
                "Cl37":      [( 17., 37., 0.931494061 ,"Cl37"      )],
                "Argon":     [( 18., 40., 0.931494061 ,"Argon"     ),
                              ( 18., 36., 0.931494061 ,"Argo36"    ),
                              ( 18., 37., 0.931494061 ,"Argo37"    ),
                              ( 18., 38., 0.931494061 ,"Argo38"    )],
                              # ( 18., 39., 0.931494061 ,"Argo39"    ),
                              # ( 18., 42., 0.931494061 ,"Argo42"    )],
                "Ar40":      [( 18., 40., 0.931494061 ,"Ar40"      )],
                "Ar36":      [( 18., 36., 0.931494061 ,"Ar36"      )],
                "Ar37":      [( 18., 37., 0.931494061 ,"Ar37"      )],
                "Ar38":      [( 18., 38., 0.931494061 ,"Ar38"      )],
                "Ar39":      [( 18., 39., 0.931494061 ,"Ar39"      )],
                "Ar42":      [( 18., 42., 0.931494061 ,"Ar42"      )],                              
                "Potassium": [( 19., 39., 0.931494061 ,"Potassium" ),
                              ( 19., 40., 0.931494061 ,"Pota40"    ),
                              ( 19., 41., 0.931494061 ,"Pota41"    )],
                "P39":       [( 19., 39., 0.931494061 ,"P39"       )],
                "P40":       [( 19., 40., 0.931494061 ,"P40"       )],
                "P41":       [( 19., 41., 0.931494061 ,"P41"       )],
                "Calcium":   [( 20., 40., 0.931494061 ,"Calcium"   ),
                              ( 20., 41., 0.931494061 ,"Calc41"    ),
                              ( 20., 42., 0.931494061 ,"Calc42"    ),
                              ( 20., 43., 0.931494061 ,"Calc43"    ),
                              ( 20., 44., 0.931494061 ,"Calc44"    ),
                              ( 20., 46., 0.931494061 ,"Calc46"    ),
                              ( 20., 48., 0.931494061 ,"Calc48"    )],
                "Ca40":      [( 20., 40., 0.931494061 ,"Ca40"      )],
                "Ca41":      [( 20., 41., 0.931494061 ,"Ca41"      )],
                "Ca42":      [( 20., 42., 0.931494061 ,"Ca42"      )],
                "Ca43":      [( 20., 43., 0.931494061 ,"Ca43"      )],
                "Ca44":      [( 20., 44., 0.931494061 ,"Ca44"      )],
                "Ca46":      [( 20., 46., 0.931494061 ,"Ca46"      )],
                "Ca48":      [( 20., 48., 0.931494061 ,"Ca48"      )],
                "Scandium":  [( 21., 45., 0.931494061 ,"Scandium"  )],
                              #( 21., 46., 0.931494061 ,"Scan46"    )],
                "Sc45":      [( 21., 45., 0.931494061 ,"Sc45"      )],
                "Sc46":      [( 21., 46., 0.931494061 ,"Sc46"      )],
                "Titanium":  [( 22., 48., 0.931494061 ,"Titanium"  ),
                              ( 22., 44., 0.931494061 ,"Tita44"    ),
                              ( 22., 46., 0.931494061 ,"Tita46"    ),
                              ( 22., 47., 0.931494061 ,"Tita47"    ),
                              ( 22., 49., 0.931494061 ,"Tita49"    ),
                              ( 22., 50., 0.931494061 ,"Tita50"    )],
                "Ti48":      [( 22., 48., 0.931494061 ,"Ti48"      )],
                "Ti44":      [( 22., 44., 0.931494061 ,"Ti44"      )],
                "Ti46":      [( 22., 46., 0.931494061 ,"Ti46"      )],
                "Ti47":      [( 22., 47., 0.931494061 ,"Ti47"      )],
                "Ti49":      [( 22., 49., 0.931494061 ,"Ti49"      )],
                "Ti50":      [( 22., 50., 0.931494061 ,"Ti50"      )],
                "Vanadium":  [( 23., 51., 0.931494061 ,"Vanadium"  ),
                              ( 23., 49., 0.931494061 ,"Vana49"    ),
                              ( 23., 50., 0.931494061 ,"Vana50"    )],
                "V51":       [( 23., 51., 0.931494061 ,"V51"       )],
                "V49":       [( 23., 49., 0.931494061 ,"V49"       )],
                "V50":       [( 23., 50., 0.931494061 ,"V50"       )],
                "Chromium":  [( 24., 52., 0.931494061 ,"Chromium"  ),
                              #( 24., 48., 0.931494061 ,"Chro48"    ),
                              ( 24., 50., 0.931494061 ,"Chro50"    ),
                              ( 24., 51., 0.931494061 ,"Chro51"    ),
                              ( 24., 53., 0.931494061 ,"Chro53"    ),
                              ( 24., 54., 0.931494061 ,"Chro54"    )],
                "Cr52":      [( 24., 52., 0.931494061 ,"Cr52"      )],
                "Cr48":      [( 24., 48., 0.931494061 ,"Cr48"      )],
                "Cr50":      [( 24., 50., 0.931494061 ,"Cr50"      )],
                "Cr51":      [( 24., 51., 0.931494061 ,"Cr51"      )],
                "Cr53":      [( 24., 53., 0.931494061 ,"Cr53"      )],
                "Cr54":      [( 24., 54., 0.931494061 ,"Cr54"      )],
                "Manganese": [( 25., 55., 0.931494061 ,"Manganese" ),
                              #( 25., 52., 0.931494061 ,"Mang52"    ),
                              ( 25., 53., 0.931494061 ,"Mang53"    ),
                              ( 25., 54., 0.931494061 ,"Mang54"    )],
                "Mn55":      [( 25., 55., 0.931494061 ,"Mn55"      )],
                "Mn52":      [( 25., 52., 0.931494061 ,"Mn52"      )],
                "Mn53":      [( 25., 53., 0.931494061 ,"Mn53"      )],
                "Mn54":      [( 25., 54., 0.931494061 ,"Mn54"      )],
                "Iron":      [( 26., 56., 0.931494061 ,"Iron"      ),
                              ( 26., 54., 0.931494061 ,"Iro54"     ),
                              ( 26., 55., 0.931494061 ,"Iro55"     ),
                              ( 26., 57., 0.931494061 ,"Iro57"     ),
                              ( 26., 58., 0.931494061 ,"Iro58"     ),
                              ( 26., 60., 0.931494061 ,"Iro60"     )],
                "Fe56":      [( 26., 56., 0.931494061 ,"Fe56"      )],
                "Fe54":      [( 26., 54., 0.931494061 ,"Fe54"      )],
                "Fe55":      [( 26., 55., 0.931494061 ,"Fe55"      )],
                "Fe57":      [( 26., 57., 0.931494061 ,"Fe57"      )],
                "Fe58":      [( 26., 58., 0.931494061 ,"Fe58"      )],
                "Fe60":      [( 26., 60., 0.931494061 ,"Fe60"      )],                              
                "Cobalt":    [( 27., 59., 0.931494061 ,"Cobalt"    ),
                              #( 27., 60., 0.931494061 ,"Coba60"    ),
                              #( 27., 56., 0.931494061 ,"Coba56"    ),
                              ( 27., 57., 0.931494061 ,"Coba57"    )],
                              #( 27., 58., 0.931494061 ,"Coba58"    )],
                "Co59":      [( 27., 59., 0.931494061 ,"Co59"      )],
                "Co60":      [( 27., 60., 0.931494061 ,"Co60"      )],
                "Co56":      [( 27., 56., 0.931494061 ,"Co56"      )],
                "Co57":      [( 27., 57., 0.931494061 ,"Co57"      )],
                "Co58":      [( 27., 58., 0.931494061 ,"Co58"      )],                              
                "Nickel":    [( 28., 58., 0.931494061 ,"Nickel"    ),
                              ( 28., 56., 0.931494061 ,"Nick56"    ),
                              ( 28., 59., 0.931494061 ,"Nick59"    ),
                              ( 28., 60., 0.931494061 ,"Nick60"    ),
                              ( 28., 61., 0.931494061 ,"Nick61"    ),
                              ( 28., 62., 0.931494061 ,"Nick62"    ),
                             # ( 28., 63., 0.931494061 ,"Nick63"    ),
                              ( 28., 64., 0.931494061 ,"Nick64"    )],
                "Ni58":      [( 28., 58., 0.931494061 ,"Ni58"      )],
                "Ni56":      [( 28., 56., 0.931494061 ,"Ni56"      )],
                "Ni59":      [( 28., 59., 0.931494061 ,"Ni59"      )],
                "Ni60":      [( 28., 60., 0.931494061 ,"Ni60"      )],
                "Ni61":      [( 28., 61., 0.931494061 ,"Ni61"      )],
                "Ni62":      [( 28., 62., 0.931494061 ,"Ni62"      )],
                "Ni63":      [( 28., 63., 0.931494061 ,"Ni63"      )],
                "Ni64":      [( 28., 64., 0.931494061 ,"Ni64"      )],
    }
def findKey(Check,dict):
    for key, value in dict.items():
        if key in Check :
            return value
    return ([])
# ========= convert to np.array =============
def To_np_array(v):
    if not isinstance(v,(np.ndarray,)):
        v = np.asarray(v)
    return v
# ========= Beta Evaluation from Tkin ==========
def beta_(T,T0):
    #if DEBUG:
      #print "BETA %f %f" %(T,T0)
    tt = T + T0
    t2 = tt + T0
    beta = np.sqrt(T*t2)/tt
    return beta    
# ========= Rigidity Evaluation from Tkin ==========
def Rigidity(T,MassNumber=1.,Z=1.):
    MassNumber=float(MassNumber)
    Z=float(Z)
    T0=0.931494061
    if np.fabs(Z)==1.:
        T0 = 0.938272046
    if MassNumber==0.:
        T0 = 5.11e-4
        MassNumber = 1.
    return MassNumber/np.fabs(Z)*np.sqrt(T*(T+2.*T0))

# ========= Tkin Evaluation from Rigidity ==========
def Energy(R,MassNumber=1.,Z=1.):
    MassNumber=float(MassNumber)
    Z=float(Z)
    T0=0.931494061
    if np.fabs(Z)==1.:
        T0 = 0.938272046
    if MassNumber==0.:
        T0 = 5.11e-4
        MassNumber = 1.
    return np.sqrt((Z*Z)/(MassNumber*MassNumber)*(R*R)+(T0*T0))-T0;

# ========= Flux conversion factor from Rigidity to Tkin ==========
def dT_dR(T=1,R=1,MassNumber=1.,Z=1.):
    MassNumber=float(MassNumber)
    Z=float(Z)
    T0=0.931494061
    if np.fabs(Z)==1.:
        T0 = 0.938272046
    if MassNumber==0.:
        T0 = 5.11e-4
        MassNumber = 1.
    return Z*Z/(MassNumber*MassNumber)*R/(T+T0)

# ========= Flux conversion factor from Tkin to Rigidity ==========
def dR_dT(T=1,R=1,MassNumber=1.,Z=1.):
    MassNumber=float(MassNumber)
    Z=float(Z)
    T0=0.931494061
    if np.fabs(Z)==1.:
        T0 = 0.938272046
    if MassNumber==0.:
        T0 = 5.11e-4
        MassNumber = 1.
    return MassNumber/np.fabs(Z)*(T+T0)/np.sqrt(T*(T+2.*T0))

# ========= Flux Conversion from Tkin --> Rigi ===================
def Tkin2Rigi_FluxConversione(Xval,Spectra,MassNumber=1.,Z=1.):
    Rigi = np.array([ Rigidity(T,MassNumber=MassNumber,Z=Z) for T in Xval ])
    Flux = np.array([ Flux*dT_dR(T=T,R=R,MassNumber=MassNumber,Z=Z) for T,R,Flux in zip(Xval,Rigi,Spectra) ])
    return (Rigi,Flux)
# ======= Linear Interpolation in log scale of vx,vy array using Newvx bins ======
def LinLogInterpolation(vx,vy,Newvx):
    vx = To_np_array(vx)
    vy = To_np_array(vy)
    Newvx = To_np_array(Newvx)
    # convert all in LogLogscale for linear interpolation
    Lvx,Lvy = np.log10(vx),np.log10(vy)
    LNewvx  = np.log10(Newvx)
    # Linear Interpolation 
    ILvy = interp1d(Lvx,Lvy,bounds_error=False,fill_value='extrapolate')(LNewvx)
    # return array in nominal scale
    return 10**ILvy
#-------------------------------------------------------------------
#--------------
def mkdir_p(path):
    import os       # OS directory manager
    import errno    # error names
    # --- this function create a folder and check if this already exist ( like mkdir -p comand)
    try:
       os.makedirs(path)
    except OSError as exc: # Python >2.5
       if exc.errno == errno.EEXIST and os.path.isdir(path):
          pass
#--------------
################### -------------------------------------- #########
import matplotlib as mpl
mpl.use('Agg')  
import matplotlib.pyplot as pl
import matplotlib.ticker
axis_font = {'fontname':'DejaVu Sans', 'size':'15.5'}
yaxis_font = {'fontname':'DejaVu Sans', 'size':'15.5'}
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rcParams['xtick.major.size']=8
mpl.rcParams['xtick.minor.size']=4
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['ytick.minor.size']=4
mpl.rcParams['axes.linewidth']=1.5
mpl.rcParams['xtick.major.width']=1.5
mpl.rcParams['xtick.minor.width']=1.5
mpl.rcParams['ytick.major.width']=1.5
mpl.rcParams['ytick.minor.width']=1.5  
mpl.rcParams['xtick.top']=True
mpl.rcParams['ytick.right']=True
mpl.rcParams['xtick.direction']="in"
mpl.rcParams['ytick.direction']="in"
#####################################################################

#-------------------------------------------------------------------
def LoadLIS(InputLISFile='../LIS/ProtonLIS_ApJ28nuclei.gz'):
    # --------- Load LIS
    ERsun=8.33
    """Open Fits File and Store particle flux in a dictionary 'ParticleFlux'
    We store it as a dictionary containing a dictionary, containing an array, where the first key is Z, the second key is A and then we have primaries, secondaries in an array
    - The option GALPROPInput select if LIS is a galprop fits file or a generic txt file
    """ 

    galdefid=InputLISFile
    hdulist = pyfits.open(galdefid)         # open fits file
    #hdulist.info()
    data = hdulist[0].data                  # assign data structure di data
    #Find out which indices to interpolate over for Rsun
    Rsun=ERsun     # Earth position in the Galaxy
    R = (np.arange(int(hdulist[0].header["NAXIS1"]))) * hdulist[0].header["CDELT1"] + hdulist[0].header["CRVAL1"]
    inds = []
    weights = []
    if (R[0] > Rsun):
      inds.append(0)
      weights.append(1)
    elif (R[-1] <= Rsun):
      inds.append(-1)
      weights.append(1)
    else:
        for i in range(len(R)-1):
            if (R[i] <= Rsun and Rsun < R[i+1]):
                inds.append(i)
                inds.append(i+1)
                weights.append((R[i+1]-Rsun)/(R[i+1]-R[i]))
                weights.append((Rsun-R[i])/(R[i+1]-R[i]))
                break
                
    # print("DEBUGLINE:: R=",R)
    # print("DEBUGLINE:: weights=",weights)
    # print("DEBUGLINE:: inds=",inds)


    # Calculate the energy for the spectral points.. note that Energy is in MeV
    energy = 10**(float(hdulist[0].header["CRVAL3"]) + np.arange(int(hdulist[0].header["NAXIS3"]))*float(hdulist[0].header["CDELT3"]))


    #Parse the header, looking for Nuclei definitions
    ParticleFlux = {}

    Nnuclei = hdulist[0].header["NAXIS4"]
    for i in range(1, Nnuclei+1):
            id = "%03d" % i
            Z = int(hdulist[0].header["NUCZ"+id])
            A = int(hdulist[0].header["NUCA"+id])
            K = int(hdulist[0].header["NUCK"+id])
            
            #print("id=%s Z=%d A=%d K=%d"%(id,Z,A,K))
            #Add the data to the ParticleFlux dictionary
            if Z not in ParticleFlux:
                    ParticleFlux[Z] = {}
            if A not in ParticleFlux[Z]:
                    ParticleFlux[Z][A] = {}
            if K not in ParticleFlux[Z][A]:
                    ParticleFlux[Z][A][K] = []
                    # data structure
                    #    - Particle type, identified by "id", the header allows to identify which particle is
                    #    |  - Energy Axis, ":" takes all elements
                    #    |  | - not used
                    #    |  | |  - distance from Galaxy center: inds is a list of position nearest to Earth position (Rsun)
                    #    |  | |  |
            d = ( (data[i-1,:,0,inds].swapaxes(0,1)) * np.array(weights) ).sum(axis=1) # real solution is interpolation between the nearest solution to Earh position in the Galaxy (inds)
            ParticleFlux[Z][A][K].append(1e7*d/energy**2) #1e7 is conversion from [cm^2 MeV]^-1 --> [m^2 GeV]^-1
            #print (Z,A,K)
            #print ParticleFlux[Z][A][K]

    ## ParticleFlux[Z][A][K] contains the particle flux for all considered species  galprop convention wants that for same combiantion of Z,A,K firsts are secondaries, latter Primary
    energy = energy/1e3 # convert energy scale from MeV to GeV
    #if A>1:
    #  energy = energy/float(A)
    hdulist.close()
    LISSpectra = [ 0 for T in energy]
    LIS_Tkin=energy
    LIS_Flux=ParticleFlux  
    return (LIS_Tkin,LIS_Flux)

def Spectra(RawMatrixFile,LIS,T0,Z,A):
#     """Return the mdulated spectra according to simualtion in HelModSimPath given the LIS in (LIS_Tkin,LIS_Flux)
#        ---------- error status ---------
#       -1: HelModSimPath file not exist
#       -2: something wrong while opening the file (maybe empty file or corrupted)
#     """ 

#     # -------------------------- Check is Simulation actually exist in the archive
#     if not os.path.isfile(RAWFilePART):
#         print("ERROR::Spectra() %s file not found"%(RAWFilePART))
#         return (-1.*np.ones(1),-1.*np.ones(1))
    LIS_Tkin,ParticleFlux=LIS

    try:
        TK_LISSpectra= (ParticleFlux[Z][A][0])[-1]   # the primary spectrum is always the last one (if exist)
    except:
        print(f"ERROR loading LIS Z={Z} A={A}")
        exit(-1)
    # include secondary spectra
    for SecInd in range(len(ParticleFlux[Z][A][0])-1):
        TK_LISSpectra= TK_LISSpectra+(ParticleFlux[Z][A][0])[SecInd] # Sum All secondaries
    LIS_Flux=TK_LISSpectra
    # ------------------------------
    # get the probability distribution function
    # ------------------------------
    try:
        # OuterEnergy_low         = data['OuterEnergy_low']
        BounduaryDistribution   = RawMatrixFile['BounduaryDistribution']
        InputEnergy             = To_np_array([ a for a in RawMatrixFile['InputEnergy']])
        NGeneratedPartcle       = To_np_array([ a for a in RawMatrixFile['NGeneratedPartcle']])
        OuterEnergy             = RawMatrixFile['OuterEnergy']
    except Exception as e: 
        print(e)
        print("ERROR::Spectra() something wrong while opening the file ",RawMatrixFile," ")
        return (-2.*np.ones(1),-2.*np.ones(1))
    
    # ------------------------------
    # Interpolate LIS
    # ------------------------------
    # print(LIS_Tkin,LIS_Flux,InputEnergy)
    ILIS = LinLogInterpolation(LIS_Tkin,LIS_Flux,InputEnergy)
    if OuterEnergy.dtype=='float64':
        OLIS = LinLogInterpolation(LIS_Tkin,LIS_Flux,OuterEnergy)
        OENK = OuterEnergy

    #print(BounduaryDistribution)
    # ------------------------------
    # Evalaute Flux
    # ------------------------------
    UnNormFlux = np.zeros(len(InputEnergy))
    for indexTDet in range(len(InputEnergy)):
        #EnergyDetector = InputEnergy[indexTDet]
        Nbind = []     # number of bin used to compute flux.
        if OuterEnergy.dtype!='float64':
            OLIS = LinLogInterpolation(LIS_Tkin,LIS_Flux,OuterEnergy[indexTDet])
            OENK = OuterEnergy[indexTDet]

        for indexTLIS in range(len(OLIS)):
            EnergyLIS = OENK[indexTLIS]
            # print(EnergyLIS)
            UnNormFlux[indexTDet]+=BounduaryDistribution[indexTDet][indexTLIS]*OLIS[indexTLIS] /beta_(EnergyLIS,T0)
            if BounduaryDistribution[indexTDet][indexTLIS]>0: Nbind.append(indexTLIS)
        # if len(Nbind)<=3 and not NewCalc:  #If the number of bin used to compute flux is 1, it means that it is a delta function. and the flux estimation could be not accurate             
        #     BDist=0
        #     for iBD in Nbind:
        #         BDist+=  BounduaryDistribution[indexTDet][iBD]
        #     UnNormFlux[indexTDet]=BDist*ILIS[indexTDet] /beta_(InputEnergy[indexTDet],T0) #this trick center the LIS estimation on the output energy since the energey distribution is smaller than Bin resolution of the matrix
    J_Mod = [ UnFlux/Npart *beta_(T,T0) for T,UnFlux,Npart in zip(InputEnergy,UnNormFlux,NGeneratedPartcle)]
    # print J_Mod
    # -- Reverse order in the List
    # EnergyBinning = np.array(InputEnergy[::-1])
    # J_Mod         = np.array(J_Mod[::-1]) 
    if InputEnergy[0]>InputEnergy[-1]:
        # -- Reverse order in the List
        EnergyBinning = np.array(InputEnergy[::-1])
        J_Mod         = np.array(J_Mod[::-1]) 
        LIS           = np.array(ILIS[::-1]) 
    else:
        EnergyBinning = np.array(InputEnergy[:])
        J_Mod         = np.array(J_Mod[:]) 
        LIS           = np.array(ILIS[:]) 
    return (EnergyBinning,J_Mod,LIS) 


## --------------------------------- MAIN CODE ---------------------------------------------------
if __name__ == "__main__":
    ArchiveDir = os.getcwd()
    mkdir_p("output")
    ################## External Commands ###############
    import sys, getopt
    arguments=sys.argv[1:]
    try:
        opts, args = getopt.getopt(arguments,"a:",[
                             "ArchivePATH=",
                             ])
        
    except getopt.GetoptError:
        HelpManual()
        sys.exit(2)
    # ----------------------------------------------------------------
    #--------------- Check arguments ---------------------------------
    for opt, arg in opts:
        if opt in ("-a","--ArchivePATH"): # set local path where is the HelMod Archive 
            ArchiveDir=arg
            
    ###################################################    

    ###### Crea lista dei file eseguibili processati nella cartella
    ## questi sono salvati in un file in EXELIST 
    ## da porre attenzione che potrebbero esserci doppioni
    ExeDirList=[] # conterrà la lista degli eseguibili (e quindi delle cartelle dei parametri create)
    for line in open(EXELIST).readlines():
        if not(line.startswith("#")):
            ExeName=line.split()[0]
            if ExeName not in ExeDirList:
                ExeDirList.append(ExeName)
    
    ###### Carica la lista delle simulazioni che dovrebbero essere presenti nella cartella
    SimList=[]
    for line in open("Simulations.list").readlines():
        if not(line.startswith("#") or 'Electron' in line or 'Positron' in line):
            SingleSim=line.replace("\t","").split("|") # carica tutte le colonne
            SimList.append(SingleSim) 
    #print(SimList)   



    ###### cicla tra le cartelle degli esperimenti per 
    for CodeVersionName in ExeDirList:
        print(f"... processing directory {CodeVersionName}")
        ThisSimulation="%s/%s"%(ArchiveDir,CodeVersionName)
            
        
        ###### all'interno della cartella degli eseguibili ci sono le cartelle con le simulazioni
        ## le simulazioni sono composte poi da input files uno per ogni isotopo da simulare.
        ## l'output finale contiene tutti gli isotopi
        # - apri Simulation List    

        for iNSim in range(len(SimList)):
            SimName  = SimList[iNSim][0].strip()
            Ion      = SimList[iNSim][1].strip()
            FileName = SimList[iNSim][2].strip()
            TKO= False if ("Rigidity" in FileName) else True
            Ion=Ion.strip()
            InputDirPATH=f"{ThisSimulation}/{SimName}"
            OutputFile=f"{InputDirPATH}/K0Scan_{Ion}{'_TKO' if TKO  else ''}_RawMatrixFile.pkl"
            print(f"... processing Ion {Ion}")
            # se non esiste salta la cartella
            if not os.path.exists(InputDirPATH): 
                print(f"dir {InputDirPATH} not found")
                continue
            # se esiste già output salta la cartella
            if (not os.path.isfile(OutputFile)):
                print(f"{Ion} pkl file not found")
                continue

            # carica il dizionario con le matrici finali della simulazione
            RawMatrixFile={}
            try:
                with open(OutputFile, "rb") as f:
                     RawMatrixFile=pickle.load(f)
            except:
                print(f"Some error occours while opening {OutputFile}")
                continue

            # carica dati sperimentali
            if "CR" in SimName:
              ExpEnRig=np.loadtxt(f"{ArchiveDir}/DataTXT/{FileName.strip()}",unpack=True)
              ExpFlux=np.zeros_like(ExpEnRig)
              Exp_Error=[np.zeros_like(ExpEnRig),np.zeros_like(ExpEnRig)]
            else:
              ExpEnRig,ExpFlux,ExpErrInf,ExpErrSup=np.loadtxt(f"{ArchiveDir}/DataTXT/{FileName.strip()}",unpack=True)
              Exp_Error=[ExpErrInf,ExpErrSup]

            # - ottieni la lista degli isotopi da simulare
            IsotopesList=findKey(Ion,Isotopes_dict)
            print(IsotopesList)
            if len(IsotopesList)<=0: 
                print(f"################################################################")
                print(f"WARNING:: {Ion} not found in Isotopes_dict, please Check")
                print(f"################################################################")
                continue
            # carica il file del LIS
            LISPATH=f"{LISPATH_root}{LISPATH_dict[Ion]}"
            print(f"... LIS {LISPATH_dict[Ion]}")
            # carica i file di output e somma eventuali risultati doppi

            print(RawMatrixFile.keys())

            K0vals=list(RawMatrixFile.keys())
            for K0 in K0vals:
              SimEnRig=np.zeros_like(ExpEnRig)
              SimFlux =np.zeros_like(ExpEnRig)
              SimLIS  =np.zeros_like(ExpEnRig)
              for iIsot  in range(len(IsotopesList)) :
                  Z,A,T0,Isotope = IsotopesList[iIsot]
                  print(Isotope)
                  LIS=LoadLIS(LISPATH)
                  #print(LIS)
                  EnergyBinning,J_Mod,J_LIS=Spectra(RawMatrixFile[K0][Isotope],LIS,T0,Z,A)
                  #print(J_LIS,J_Mod)
                  if TKO:
                      SimEnRig=EnergyBinning
                  else:
                      SimEnRig,J_Mod=Tkin2Rigi_FluxConversione(EnergyBinning,J_Mod,MassNumber=A,Z=Z)
                      SimEnRig,J_LIS=Tkin2Rigi_FluxConversione(EnergyBinning,J_LIS,MassNumber=A,Z=Z)
                  
                  SimFlux+=J_Mod
                  SimLIS+=J_LIS
              # # if "Fluorine" in Ion:
              # #     SimFlux*=0.896
              # #     SimLIS*=0.896
              # if "ACECRIS" in SimName:
              #     Lmin,Lmax=1e-1,1e0  
              # else:
              #     Lmin,Lmax=1,3e1                                 # x limits
              # if ExpEnRig.min()<=Lmin : Lmin=0.5*ExpEnRig.min()
              # if ExpEnRig.max()>=Lmax : Lmax=2*ExpEnRig.max()
              # Limit=[Lmin, Lmax]
              # fig = pl.figure(figsize=(7, 8))
              # ## Pannello superiore
              # ax0 = pl.subplot2grid((3, 1), (0, 0), rowspan=2)
              # ax0.plot(SimEnRig,SimLIS,'--k',  linewidth=2,label=r"LIS") 
              # ax0.plot(SimEnRig,SimFlux,'-',color=HMColorLine, linewidth=2,label="Modulated Spectra")
              # #ax0.fill_between(Best_X, MinSimTot, MaxSimTot,facecolor='grey',alpha=0.5)
              # #
              # ax0.errorbar(ExpEnRig, ExpFlux, yerr=Exp_Error, label="%s" %(SimName),marker=".",color='k',linestyle='none')
              # pl.legend(numpoints=1,fancybox=True,fontsize=14,loc="lower left")
              # pl.yscale('log')
              # pl.xscale('log')    
              # ax0.set_xticklabels([])  
              # if TKO:
              #     pl.xlim(Limit)
              #     pl.ylabel("Differential Intensity "+r"[(m$^2$ s sr GeV/nuc)$^{-1}$]",**axis_font)    
              # else:
              #     pl.xlim(Limit)
              #     pl.ylabel("Differential Intensity "+r" [(m$^2$ s sr GV)$^{-1}$]",**axis_font)
              # ## Pannello inferiore
              # ax1 = pl.subplot2grid((3, 1), (2, 0))
              #     # DistanceFlux    = np.zeros(len(ExpEnRig))
              #     # expDist         = (ExpFlux-SimFlux)/SimFlux
              #     # expDistErr      = Exp_Error/SimFlux
              #     # DistanceLIS     = (SimLIS-SimFlux)/SimFlux
              #     # #MinDisTot       = (MinSimTot-BestSpectra_S)/BestSpectra_S
              #     # #MaxDisTot       = (MaxSimTot-BestSpectra_S)/BestSpectra_S
              # DistanceFlux    = (SimFlux-ExpFlux)/ExpFlux
              # expDist         = np.zeros(len(ExpEnRig)) 
              # expDistErr      = Exp_Error/ExpFlux
              # DistanceLIS     = (SimLIS-ExpFlux)/ExpFlux

              # ax1.plot(SimEnRig, DistanceLIS, '--k')
              # ax1.errorbar(ExpEnRig,expDist, yerr=expDistErr, marker=".",color='k',linestyle='none')
              # ax1.plot(ExpEnRig, DistanceFlux, "-",color=HMColorLine)
              # #ax1.fill_between(Best_X, MinDisTot, MaxDisTot,facecolor='grey',alpha=0.5)
              # #
              # pl.ylim([-0.39,0.39])
              # if TKO:
              #     pl.xlim(Limit)
              #     pl.xlabel("Kinetic Energy [GeV/nuc]",**axis_font)
              # else:
              #     pl.xlim(Limit)
              #     pl.xlabel("Rigidity [GV]",**axis_font)
              # pl.ylabel("Relative difference",**axis_font)
              # pl.xscale('log')
              # ## settaggi finali
              # pl.tight_layout()
              # pl.subplots_adjust( wspace=0, hspace=0.,
              #                # left = 0.13  ,# the left side of the subplots of the figure
              #                #  right = 0.97   ,# the right side of the subplots of the figure
              #                # bottom = 0.09  ,# the bottom of the subplots of the figure
              #                #  top = 0.98     # the top of the subplots of the figure
              #                 )
              # fig.align_ylabels()
              # pl.savefig("../1_OutputFigures/Figure_%s_%s.png" %(SimName,Ion), dpi=100)
              # pl.close()
              K0string=f"{K0:.6e}".replace('.',"_")
              np.savetxt("output/ModK0Flux_%s_%s_%s_%s.dat" %(K0string,CodeVersionName,SimName,Ion),np.c_[SimEnRig,SimFlux,SimLIS,ExpFlux,Exp_Error[0],Exp_Error[1]])




        #### Close and Clean
 







    exit()

