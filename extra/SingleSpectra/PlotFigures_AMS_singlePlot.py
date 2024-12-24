# -*- coding: utf-8 -*-
# developed: Nov 2023
# last update: Nov 2023
''' 
description: comparison of the calculatedspectra with AMS-02
Shown also the relative difference between our calculations and the data sets. The dashed black lines show the
LIS, and the solid red lines are the corresponding modulated spectra.

'''
# -- libraries
import numpy as np
import os                                 # OS directory manager
import glob
import pickle
import astropy.io.fits as pyfits # per elaborare i fits file
from scipy import interpolate
from scipy.interpolate import interp1d # interpolazione
# -- directory e nomi file generali
ArchiveSim = "../Flux_Confirmed/AMS-02/"
LISPATH_root="../LISs/"
LIS_1 = "all_nuclei_sets_nuclei_57_All_2023_final"        # 1) final è il set completo comune ad all_nuclei con He3 primario
LIS_2 = "all_nuclei_sets_nuclei_57_All_2023_Dbreak"       # 2) Dbreak è il set con break diffusivo alternativo al break nell'iniezione
LabLIS = r"GALPROP LIS"
LabLIS2 = r"GALPROP LIS with diff. break"
LISPATH=f"{LISPATH_root}{LIS_1}"
LISPATH2=f"{LISPATH_root}{LIS_2}"

SimList = [
    ("AMS-02_PRL2019"       ,"He3"        , "Rigidity_AMS-02_PRL2019_He3.dat"         	, "2011", "2017"	,"1","0","0", "short_ref:PRL 123 (2019) 181102 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.123.181102 "),
    ("AMS-02_PRL2019"       ,"He4"        , "Rigidity_AMS-02_PRL2019_He4.dat"         	, "2011", "2017"	,"1","0","0", "short_ref:PRL 123 (2019) 181102 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.123.181102 "),
    ("AMS-02_PRL2021"       ,"Fluorine"   , "Rigidity_AMS-02_PRL2021_Fluorine.dat"    	, "2011", "2019"	,"1","0","0", "short_ref:PRL 126 (2021) 081102 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.126.081102 "),
    ("AMS-02_PRL2021"       ,"Iron"       , "Rigidity_AMS-02_PRL2021_Iron.dat"        	, "2011", "2019"	,"1","0","0", "short_ref:PRL 126 (2021) 041104 "," DOI: http://doi.org/10.1103/PhysRevLett.126.041104 "),
    ("AMS-02_PRL2021"       ,"Sodium"     , "Rigidity_AMS-02_PRL2021_Sodium.dat"      	, "2011", "2019"	,"1","0","0", "short_ref:PRL 127 (2021) 021101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.127.021101 "),
    ("AMS-02_PRL2021"       ,"Aluminum"   , "Rigidity_AMS-02_PRL2021_Aluminum.dat"    	, "2011", "2019"	,"1","0","0", "short_ref:PRL 127 (2021) 021101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.127.021101 "),
    ("AMS-02_PRL2018"       ,"Beryllium"  , "Rigidity_AMS-02_PRL2018_Beryllium.dat"   	, "2011", "2016"	,"1","0","0", "short_ref:PRL 120 (2018) 021101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.120.021101 "),
    ("AMS-02_PRL2018"       ,"Boron"      , "Rigidity_AMS-02_PRL2018_Boron.dat"       	, "2011", "2016"	,"1","0","0", "short_ref:PRL 120 (2018) 021101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.120.021101 "),
    ("AMS-02_PRL2017"       ,"Carbon"     , "Rigidity_AMS-02_PRL2017_Carbon.dat"      	, "2011", "2016"	,"1","0","0", "short_ref:PRL 119 (2017) 251101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.119.251101 "),
    ("AMS-02_PRL2017"       ,"Helium"     , "Rigidity_AMS-02_PRL2017_Helium.dat"      	, "2011", "2016"	,"1","0","0", "short_ref:PRL 119 (2017) 251101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.119.251101 "),
    ("AMS-02_PRL2015"       ,"Helium"     , "Rigidity_AMS-02_PRL2015_Helium.dat"      	, "2011", "2013"	,"1","0","0", "short_ref:PRL 115 (2015) 211101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.115.211101 "),
    ("AMS-02_PRL2018"       ,"Lithium"    , "Rigidity_AMS-02_PRL2018_Lithium.dat"     	, "2011", "2016"	,"1","0","0", "short_ref:PRL 120 (2018) 021101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.120.021101 "),
    ("AMS-02_PRL2020"       ,"Magnesium"  , "Rigidity_AMS-02_PRL2020_Magnesium.dat"   	, "2011", "2018"	,"1","0","0", "short_ref:PRL 124 (2020) 211102 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.124.211102 "),
    ("AMS-02_PRL2020"       ,"Neon"       , "Rigidity_AMS-02_PRL2020_Neon.dat"        	, "2011", "2018"	,"1","0","0", "short_ref:PRL 124 (2020) 211102 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.124.211102 "),
    ("AMS-02_PRL2018"       ,"Nitrogen"   , "Rigidity_AMS-02_PRL2018_Nitrogen.dat"    	, "2011", "2016"	,"1","0","0", "short_ref:PRL 121 (2018) 051103 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.121.051103 "),
    ("AMS-02_PRL2017"       ,"Oxygen"     , "Rigidity_AMS-02_PRL2017_Oxygen.dat"      	, "2011", "2016"	,"1","0","0", "short_ref:PRL 119 (2017) 251101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.119.251101 "),
    ("AMS-02_PRL2016"       ,"Antiproton" , "Rigidity_AMS-02_PRL2016_Antiproton.dat"  	, "2011", "2015"	,"1","0","0", "short_ref:PRL 117 (2016) 091103 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.117.091103 "),
    ("AMS-02_PRL2015"       ,"Proton"     , "Rigidity_AMS-02_PRL2015_Proton.dat"      	, "2011", "2013"	,"1","0","0", "short_ref:PRL 114 (2015) 171103 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.114.171103 "),
    ("AMS-02_PRL2020"       ,"Silicon"    , "Rigidity_AMS-02_PRL2020_Silicon.dat"     	, "2011", "2018"	,"1","0","0", "short_ref:PRL 124 (2020) 211102 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.124.211102 "),
    ("AMS-02_PhysRep2021"   ,"Lithium"    , "Rigidity_AMS-02_PhysRep2021_Lithium.dat" 	, "2011", "2018"	,"1","0","0", "short_ref:Physics Reports 894 (2021) 1 "," DOI: https://dx.doi.org/10.1016/j.physrep.2020.09.003 "),
    ("AMS-02_PhysRep2021"   ,"Beryllium"  , "Rigidity_AMS-02_PhysRep2021_Beryllium.dat"	, "2011", "2018"	,"1","0","0", "short_ref:Physics Reports 894 (2021) 1 "," DOI: https://dx.doi.org/10.1016/j.physrep.2020.09.003 "),
    ("AMS-02_PhysRep2021"   ,"Boron"      , "Rigidity_AMS-02_PhysRep2021_Boron.dat"   	, "2011", "2018"	,"1","0","0", "short_ref:Physics Reports 894 (2021) 1 "," DOI: https://dx.doi.org/10.1016/j.physrep.2020.09.003 "),
    ("AMS-02_PhysRep2021"   ,"Nitrogen"   , "Rigidity_AMS-02_PhysRep2021_Nitrogen.dat"	, "2011", "2018"	,"1","0","0", "short_ref:Physics Reports 894 (2021) 1 "," DOI: https://dx.doi.org/10.1016/j.physrep.2020.09.003 "),
    ("AMS-02_PhysRep2021"   ,"Proton"     , "Rigidity_AMS-02_PhysRep2021_Proton.dat"  	, "2011", "2018"	,"1","0","0", "short_ref:Physics Reports 894 (2021) 1 "," DOI: https://dx.doi.org/10.1016/j.physrep.2020.09.003 "),
    ("AMS-02_PhysRep2021"   ,"Antiproton" , "Rigidity_AMS-02_PhysRep2021_Antiproton.dat", "2011", "2018"	,"1","0","0", "short_ref:Physics Reports 894 (2021) 1 "," DOI: https://dx.doi.org/10.1016/j.physrep.2020.09.003 "),
    ("AMS-02_PhysRep2021"   ,"Helium"     , "Rigidity_AMS-02_PhysRep2021_Helium.dat"  	, "2011", "2018"	,"1","0","0", "short_ref:Physics Reports 894 (2021) 1 "," DOI: https://dx.doi.org/10.1016/j.physrep.2020.09.003 "),
    ("AMS-02_PhysRep2021"   ,"Carbon"     , "Rigidity_AMS-02_PhysRep2021_Carbon.dat"  	, "2011", "2018"	,"1","0","0", "short_ref:Physics Reports 894 (2021) 1 "," DOI: https://dx.doi.org/10.1016/j.physrep.2020.09.003 "),
    ("AMS-02_PhysRep2021"   ,"Oxygen"     , "Rigidity_AMS-02_PhysRep2021_Oxygen.dat"  	, "2011", "2018"	,"1","0","0", "short_ref:Physics Reports 894 (2021) 1 "," DOI: https://dx.doi.org/10.1016/j.physrep.2020.09.003 "),
    ("AMS-02_PRL2023"       ,"Sulfur"     , "Rigidity_AMS-02_PRL2023_Sulfur.dat"      	, "2011", "2021"	,"1","0","0", "short_ref:PRL 130 (2023) 211002 "," DOI:  https://dx.doi.org/10.1103/PhysRevLett.130.211002 "),
    ("AMS-02_48bin_PRL2023" ,"Oxygen"     , "Rigidity_AMS-02_48bin_PRL2023_Oxygen.dat"	, "2011", "2021"	,"1","0","0", "short_ref:PRL 130 (2023) 211002 "," DOI:  https://dx.doi.org/10.1103/PhysRevLett.130.211002 "),
    ("AMS-02_PRL2023"       ,"Neon"       , "Rigidity_AMS-02_PRL2023_Neon.dat"        	, "2011", "2021"	,"1","0","0", "short_ref:PRL 130 (2023) 211002 "," DOI:  https://dx.doi.org/10.1103/PhysRevLett.130.211002 "),
    ("AMS-02_PRL2023"       ,"Magnesium"  , "Rigidity_AMS-02_PRL2023_Magnesium.dat"   	, "2011", "2021"	,"1","0","0", "short_ref:PRL 130 (2023) 211002 "," DOI:  https://dx.doi.org/10.1103/PhysRevLett.130.211002 "),
    ("AMS-02_PRL2023"       ,"Silicon"    , "Rigidity_AMS-02_PRL2023_Silicon.dat"     	, "2011", "2021"	,"1","0","0", "short_ref:PRL 130 (2023) 211002 "," DOI:  https://dx.doi.org/10.1103/PhysRevLett.130.211002 "),
    ("AMS-02_PRL2023"       ,"Fluorine"   , "Rigidity_AMS-02_PRL2023_Fluorine.dat"    	, "2011", "2021"	,"1","0","0", "short_ref:PRL 130 (2023) 211002 "," DOI:  https://dx.doi.org/10.1103/PhysRevLett.130.211002 "),
    ("AMS-02_66bin_PRL2023" ,"Oxygen"     , "Rigidity_AMS-02_66bin_PRL2023_Oxygen.dat"	, "2011", "2021"	,"1","0","0", "short_ref:PRL 130 (2023) 211002 "," DOI:  https://dx.doi.org/10.1103/PhysRevLett.130.211002 "),
    ("AMS-02_PRL2023"       ,"Boron"      , "Rigidity_AMS-02_PRL2023_Boron.dat"       	, "2011", "2021"	,"1","0","0", "short_ref:PRL 130 (2023) 211002 "," DOI:  https://dx.doi.org/10.1103/PhysRevLett.130.211002 "),
    ("AMS-02_PRL2023"       ,"Carbon"     , "Rigidity_AMS-02_PRL2023_Carbon.dat"      	, "2011", "2021"	,"1","0","0", "short_ref:PRL 130 (2023) 211002 "," DOI:  https://dx.doi.org/10.1103/PhysRevLett.130.211002 "),
    # ("AMS-02_PRL2019"       ,"He3"        , "KinEnergy_AMS-02_PRL2019_He3.dat"        	, "2011", "2017"	,"1","0","0", "short_ref:PRL 123 (2019) 181102 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.123.181102 "),
    # ("AMS-02_PRL2019"       ,"He4"        , "KinEnergy_AMS-02_PRL2019_He4.dat"        	, "2011", "2017"	,"1","0","0", "short_ref:PRL 123 (2019) 181102 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.123.181102 "),
    # ("AMS-02_PRL2018"       ,"Beryllium"  , "KinEnergy_AMS-02_PRL2018_Beryllium.dat"  	, "2011", "2016"	,"1","0","0", "short_ref:PRL 120 (2018) 021101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.120.021101 "),
    # ("AMS-02_PRL2018"       ,"Boron"      , "KinEnergy_AMS-02_PRL2018_Boron.dat"      	, "2011", "2016"	,"1","0","0", "short_ref:PRL 120 (2018) 021101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.120.021101 "),
    # ("AMS-02_PRL2017"       ,"Carbon"     , "KinEnergy_AMS-02_PRL2017_Carbon.dat"     	, "2011", "2016"	,"1","0","0", "short_ref:PRL 119 (2017) 251101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.119.251101 "),
#    ("AMS-02_PRL2019"       ,"Positron"   , "KinEnergy_AMS-02_PRL2019_Positron.dat"   	, "2011", "2017"	,"1","0","0", "short_ref:PRL 122 (2019) 041102 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.122.041102 "),
#    ("AMS-02_PRL2019"       ,"Electron"   , "KinEnergy_AMS-02_PRL2019_Electron.dat"   	, "2011", "2017"	,"1","0","0", "short_ref:PRL 122 (2019) 101101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.122.041102 "),
#    ("AMS-02_PRL2014"       ,"Electron"   , "KinEnergy_AMS-02_PRL2014_Electron.dat"   	, "2011", "2013"	,"1","0","0", "short_ref:PRL 113 (2014) 121102 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.113.121102 "),
    # ("AMS-02_PRL2017"       ,"Helium"     , "KinEnergy_AMS-02_PRL2017_Helium.dat"     	, "2011", "2016"	,"1","0","0", "short_ref:PRL 119 (2017) 251101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.119.251101 "),
    # ("AMS-02_PRL2018"       ,"Lithium"    , "KinEnergy_AMS-02_PRL2018_Lithium.dat"    	, "2011", "2016"	,"1","0","0", "short_ref:PRL 120 (2018) 021101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.120.021101 "),
    # ("AMS-02_PRL2017"       ,"Oxygen"     , "KinEnergy_AMS-02_PRL2017_Oxygen.dat"     	, "2011", "2016"	,"1","0","0", "short_ref:PRL 119 (2017) 251101 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.119.251101 "),
#    ("AMS-02_PRL2014"       ,"Positron"   , "KinEnergy_AMS-02_PRL2014_Positron.dat"   	, "2011", "2013"	,"1","0","0", "short_ref:PRL 113 (2014) 121102 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.113.121102 "),
    # ("AMS-02_PRL2015"       ,"Proton"     , "KinEnergy_AMS-02_PRL2015_Proton.dat"     	, "2011", "2013"	,"1","0","0", "short_ref:PRL 114 (2015) 171103 "," DOI: https://dx.doi.org/10.1103/PhysRevLett.114.171103 "),
#    ("AMS-02_PhysRep2021"   ,"Positron"   , "KinEnergy_AMS-02_PhysRep2021_Positron.dat"	, "2011", "2018"	,"1","0","0", "short_ref:Physics Reports 894 (2021) 1 "," DOI: https://dx.doi.org/10.1016/j.physrep.2020.09.003 "),
#    ("AMS-02_PhysRep2021"   ,"Electron"   , "KinEnergy_AMS-02_PhysRep2021_Electron.dat"	, "2011", "2018"	,"1","0","0", "short_ref:Physics Reports 894 (2021) 1 "," DOI: https://dx.doi.org/10.1016/j.physrep.2020.09.003 "),  
]
#AMSDataLabel="AMS-02 (2011-2021)"
ExeDirList = ["HelMod-4-CUDA_v3"]
HMColorLine='tab:red'
StyleLIS = "--"
StyleLIS2 = "-"
AMSmarker_prop = dict( linestyle='none', marker='o', fillstyle='full' , markersize=5, markerfacecoloralt='gray', color='k')
g=2.7
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

def GetLIS(LIS,Z, A, K=0, IncludeSecondaries=True):
  """Return the LIS for species,
      mode define the kind of LIS are interested to get according to
      0: specific Z,A LIS in Kinetic Energy per nucleon
      1: specific Z,A LIS in Rigidity
      2: Summing all isotopes with same Z and same Kinetic Energy per nucleon
      3: Summing all isotopes with same Z and same Rigidity
             return (Xbin, LIS)
             ---------- error return status ---------
            -1: Z does not exist in Galprop Fits
            -2: A does not exist in Galprop Fits
            -3: unrecognized Mode
  """
  # -------- init variables
  TK_bin,ParticleFlux=LIS


  # == check of A is available
  if (A not in ParticleFlux[Z]):     # Return -2 if A does not exist for selected Z
    return (-2.*np.ones(1),-2.*np.ones(1))
  # ==
  TK_LISSpectra= (ParticleFlux[Z][A][K])[-1]   # the primary spectrum is always the last one (if exist)
  if IncludeSecondaries:                            # include secondary spectra
      for SecInd in range(len(ParticleFlux[Z][A][K])-1):
          TK_LISSpectra= TK_LISSpectra+(ParticleFlux[Z][A][K])[SecInd] # Sum All secondaries
  return (TK_bin, TK_LISSpectra) 

#################################################################################################
#################################################################################################
## --------------------------------- MAIN CODE ---------------------------------------------------
if __name__ == "__main__":
  ArchiveDir      = ArchiveSim
  CodeVersionName = ExeDirList[0]

  print(f"... processing directory {CodeVersionName}")
  ThisSimulation="%s/%s"%(ArchiveDir,CodeVersionName)
  for ThisSim in SimList:
    ######################################################################
    ######################################################################  
    SimName  = ThisSim[0].strip()
    Ion      = ThisSim[1].strip()
    FileName = ThisSim[2].strip()
    AMSDataLabel = f"AMS-02 ({ThisSim[3].strip()}-{ThisSim[4].strip()})"
    TKO= False if ("Rigidity" in FileName) else True
    Ion=Ion.strip()
    InputDirPATH=f"{ThisSimulation}/{SimName}"
    OutputFile=f"{InputDirPATH}/{Ion}{'_TKO' if TKO  else ''}_RawMatrixFile.pkl"
    print(f"... processing Ion {Ion}")
    
    # se non esiste dai errore
    if not os.path.exists(InputDirPATH) or not os.path.isfile(OutputFile): 
        print(f"dir of file  not found {OutputFile}")
        exit()
    # carica il dizionario con le matrici finali della simulazione
    RawMatrixFile={}
    try:
        with open(OutputFile, "rb") as f:
            RawMatrixFile=pickle.load(f)
    except:
        print(f"Some error occours while opening {OutputFile}")
        exit()
    # carica dati sperimentali
    ExpEnRig,ExpFlux,ExpErrInf,ExpErrSup=np.loadtxt(f"{ArchiveDir}/DataTXT/{FileName.strip()}",unpack=True)
    Exp_Error=[ExpErrInf,ExpErrSup]
    #
    IsotopesList=findKey(Ion,Isotopes_dict)
    print(IsotopesList)
    if len(IsotopesList)<=0: 
        print(f"################################################################")
        print(f"WARNING:: {Ion} not found in Isotopes_dict, please Check")
        print(f"################################################################")
        exit()

    ######################################################################
    ######################################################################  


    ######################################################################
    ######################################################################  
    # -- 
    print(f"... {LabLIS}")
    LIS=LoadLIS(LISPATH)
    LIS2=LoadLIS(LISPATH2)
    #He3
    # carica i file di output e somma eventuali risultati doppi
    SimEnRig=np.zeros_like(ExpEnRig)
    SimFlux =np.zeros_like(ExpEnRig)
    SimLIS  =np.zeros_like(ExpEnRig)
    print(RawMatrixFile.keys())
    for iIsot  in range(len(IsotopesList)) :
        Z,A,T0,Isotope = IsotopesList[iIsot]
        print(Isotope)
        
        #print(LIS)
        EnergyBinning,J_Mod,J_LIS=Spectra(RawMatrixFile[Isotope],LIS,T0,Z,A)
        #print(J_LIS,J_Mod)
        if TKO:
            SimEnRig=EnergyBinning
        else:
            SimEnRig,J_Mod=Tkin2Rigi_FluxConversione(EnergyBinning,J_Mod,MassNumber=A,Z=Z)
            SimEnRig,J_LIS=Tkin2Rigi_FluxConversione(EnergyBinning,J_LIS,MassNumber=A,Z=Z)
        
        SimFlux+=J_Mod
        SimLIS+=J_LIS
    
    #LIS
    LISx,LISy=[],[]
    for iIsot  in range(len(IsotopesList)) :
        Z,A,T0,Isotope = IsotopesList[iIsot]
        tLISx,tLISy=GetLIS(LIS,Z,A)
        if len(LISx)==0:
            LISx,LISy=tLISx.copy(),tLISy.copy()
            if not TKO:
                LISx,LISy=Tkin2Rigi_FluxConversione(LISx,LISy,MassNumber=A,Z=Z)
        else:
            if not TKO: 
                rtLISx,rtLISy=Tkin2Rigi_FluxConversione(tLISx,tLISy,MassNumber=A,Z=Z)
                rtLISy=interpolate.interp1d(rtLISx,rtLISy,fill_value="extrapolate")(LISx)
                LISy+=rtLISy    
            else:
                LISy+=tLISy
    # second LIS
    LISx2,LISy2=[],[]
    for iIsot  in range(len(IsotopesList)) :
        Z,A,T0,Isotope = IsotopesList[iIsot]
        tLISx2,tLISy2=GetLIS(LIS2,Z,A)
        if len(LISx2)==0:
            LISx2,LISy2=tLISx2.copy(),tLISy2.copy()
            if not TKO:
                LISx2,LISy2=Tkin2Rigi_FluxConversione(LISx2,LISy2,MassNumber=A,Z=Z)
        else:
            if not TKO: 
                rtLISx2,rtLISy2=Tkin2Rigi_FluxConversione(tLISx2,tLISy2,MassNumber=A,Z=Z)
                rtLISy2=interpolate.interp1d(rtLISx2,rtLISy2,fill_value="extrapolate")(LISx2)
                LISy2+=rtLISy2    
            else:
                LISy2+=tLISy2  
    #-------------------------------------
    # normalization corrections
    if Ion=="Fluorine":
        nf=0.896
        SimFlux*=nf
        LISy*=nf
        LISy2*=nf
        SimLIS*=nf
    #--------------------------------------
    # -- DISEGNA lo scheletro del grafico
    fig = pl.figure(figsize=(7, 8.5))
    # x limits
    Lmin=1
    Lmax=1e4
    Limit=[Lmin, Lmax]
    ## Pannello superiore 
    ax0 = pl.subplot2grid((4, 1), (0, 0), rowspan=3)
    pl.yscale('log')
    pl.xscale('log')
    ax0.set_xticklabels([])  
    pl.xlim(Limit)
    pl.ylim([0.2*np.min(SimFlux*(SimEnRig**g)),5*np.max(SimFlux*(SimEnRig**g))])
    pl.ylabel("Diff. Intensity "+r"$\times$ R$^{%.1f}$ [(m$^2$ s sr)$^{-1}$(GV)$^{%.1f}$]"%(g,g-1),**axis_font)
    ## Pannello centrale
    ax1 = pl.subplot2grid((4, 1), (3, 0))
    pl.ylim([-0.39,0.39])
    pl.xlim(Limit)
    pl.xlabel("Rigidity [GV]",**axis_font)
    pl.ylabel("Relative Difference",**axis_font)
    pl.xscale('log')
    #--------------------------------------
    # -- valori da disegnare
    ## 
    ax0.plot(LISx,LISy*(LISx**g),'%s'%(StyleLIS), color='k', linewidth=2,label=r"%s"%(LabLIS)) 
    ax0.plot(LISx2[110:],LISy2[110:]*(LISx2[110:]**g),'%s'%(StyleLIS2), color='k', linewidth=2,label=r"%s"%(LabLIS2)) 
    #ax0.plot(SimEnRig,SimLIS*(SimEnRig**g),'%s'%(StyleLIS), color='k', linewidth=2,label=r"%s"%(LabLIS)) 
    ax0.plot(SimEnRig,SimFlux*(SimEnRig**g),'-',color=HMColorLine, linewidth=2,label="Modulated Spectra")
    #
    ax0.errorbar(ExpEnRig, ExpFlux*(ExpEnRig**g), yerr=Exp_Error*(ExpEnRig**g), label=r"%s %s"%(Ion,AMSDataLabel),**AMSmarker_prop)
    DistanceFlux    = np.zeros(len(SimFlux))
    expZero         = (ExpFlux-SimFlux)/SimFlux
    expZeroErr      = Exp_Error/SimFlux
    DistanceLIS     = (SimLIS-SimFlux)/SimFlux
    ax1.plot(SimEnRig, DistanceLIS, '%s'%(StyleLIS), color='k')
    ax1.errorbar(ExpEnRig,expZero, yerr=expZeroErr, **AMSmarker_prop)
    ax1.plot(SimEnRig, DistanceFlux, "-",color=HMColorLine)
    #ax1.text(0.1, 0.1, r"$^2$H", fontsize=15.5,transform=ax1.transAxes)  
    
    #--------------------------------------
    # -- settaggi finali
    ax0.legend(numpoints=1,fancybox=True,fontsize=13,loc="lower right")
    pl.tight_layout()
    pl.subplots_adjust( wspace=0, hspace=0.,
                    # left = 0.13  ,# the left side of the subplots of the figure
                    #  right = 0.97   ,# the right side of the subplots of the figure
                    # bottom = 0.09  ,# the bottom of the subplots of the figure
                    #  top = 0.98     # the top of the subplots of the figure
                    )
    fig.align_ylabels()
    pl.savefig(f"Figure_{Ion}_{SimName}.png", dpi=300)

    pl.close()
    ########################################################################
    ########################################################################
    if 'Proton' in Ion : np.savetxt(f"Proton_{SimName}.txt",np.c_[SimEnRig,SimFlux,SimLIS])
  
  exit()

