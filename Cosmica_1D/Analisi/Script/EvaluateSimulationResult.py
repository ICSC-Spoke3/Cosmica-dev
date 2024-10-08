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
# -- directory e nomi file generali

FORCE_RECALC=True # forza il ricalcolo dei file di output (if false -> skip Ions if RawMatrix File already exist)
SINGLE_ISOTOPE = True
NHELIOSPHEREREGIONS=15

def init_sims_path(version, sim_list_name, HelMod_opt=False):
    #### Directory del codice sorgente
    SOURCECODEDIR = "/home/nfsdisk/DevGC/Cosmica_1D"
    if HelMod_opt:
        SOURCECODEDIR += f"/HelMod-{version}"
    if version=="speedtest":
        PASTPARAMETERLIST = f"{SOURCECODEDIR}/Cosmica_speed_test/0_OutputFiles_{version}/ParameterListALL_v12.txt"
        FRCSTPARAMETERLIST = f"{SOURCECODEDIR}/Cosmica_speed_test/0_OutputFiles_{version}/Frcst_param.txt"
        SIMLIST = f"{SOURCECODEDIR}/Cosmica_speed_test/{sim_list_name}.list"
    else:
        PASTPARAMETERLIST = f"{SOURCECODEDIR}/0_OutputFiles_{version}/ParameterListALL_v12.txt"
        FRCSTPARAMETERLIST = f"{SOURCECODEDIR}/0_OutputFiles_{version}/Frcst_param.txt"
        SIMLIST = f"{SOURCECODEDIR}/{sim_list_name}.list"

    return SOURCECODEDIR, PASTPARAMETERLIST, FRCSTPARAMETERLIST, SIMLIST

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

def Energy(R,MassNumber=1.,Z=1.):
  import numpy
  T0=0.931494061
  if numpy.fabs(Z)==1:
    T0 = 0.938272046
  if MassNumber==0:
          T0 = 5.11e-4
          MassNumber = 1
  return numpy.sqrt((Z*Z)/(MassNumber*MassNumber)*(R*R)+(T0*T0))-T0

# ========= Flux conversion factor from Tkin to Rigidity ==========
def dlogR_dlogT(T, MassNumber=1.,Z=1.):
    MassNumber=float(MassNumber)
    Z=float(Z)
    T0=0.931494061
    if np.fabs(Z)==1.:
        T0 = 0.938272046
    if MassNumber==0.:
        T0 = 5.11e-4
        MassNumber = 1.
    return (T*T0)/(T+2*T0)

## ------------- Crea Output Files list ---------------
def CreateOutputList(SimName,Ion,FileName,InitDate,FinalDate,rad,lat,lon):
    # - lista di uscita
    OutputFileNameList=[]
    # - verifica se il file di ingresso è in rigidità o Energia Cinetica
    TKO= False if ("Rigidity" in FileName) else True


    # - ottieni la lista delle CR da simulare
    CRIni=HeliosphericParameters[0]
    CREnd=HeliosphericParameters[1]
    ######################################################################                
    # controlla se il periodo di integrazione rientra all'interno dei parametri temporali disponibili
    # se non lo fosse, si potrebbe cmq usare i parametri di forecast, ma questo caso va trattato (va aperta la lista FRCSTPARAMETERLIST e fatto una join con l'attuale, attenzione all'overlapping)
    if int(FinalDate)>int(CREnd[0]):
        print(f"WARNING:: End date for integration time ({FinalDate}) is after last available parameter ({int(CREnd[0])})")
        print(f"WARNING:: in questo caso la lista delle CR va integrata con la lista di forecast -- CASO DA SVILUPPARE skipped simulation")
        return []
    ######################################################################
    if int(InitDate)<int(CRIni[-NHELIOSPHEREREGIONS]):
        print(f"WARNING:: Initial date for integration time ({InitDate}) is before first available parameter ({int(CRIni[-15])}) (including 15 regions)")
        print("           --> skipped simulation")
        return []

    CRList=[]
    CRListParam=[]
    #print(f"{InitDate} - {FinalDate}")
    for iCR in range(len(CRIni)):
        #print(f"{int(CRIni[iCR])} {int(FinalDate)} {int(CREnd[iCR])} {int(InitDate)}")
        # se sono in un range di CR tra InitDate e FinalDate 
        if int(CRIni[iCR])<=int(FinalDate) and int(CREnd[iCR])>int(InitDate):
          CRList.append(CRIni[iCR])

        # se sono in un range di CR tra InitDate(-numero regioni) e FinalDate 
        if int(CRIni[iCR])<=int(FinalDate) and int(CREnd[iCR-(NHELIOSPHEREREGIONS-1)])>int(InitDate):
          CRListParam.append(CRIni[iCR])

        # se sono ad un CR antecedente InitDate(-numero regioni)
        if iCR-(NHELIOSPHEREREGIONS-1)>=0:
          if int(CRIni[iCR-(NHELIOSPHEREREGIONS-1)])<int(InitDate):
            break
       
        
    if len(CRList)<=0:
        print("WARNING:: CR list empty")
        return []
    for Ion in Ions:
        # - ottieni la lista degli isotopi da simulare
        IsotopesList=findKey(Ion,Isotopes_dict)
        #print(IsotopesList)
        if len(IsotopesList)<=0: 
            print(f"################################################################")
            print(f"WARNING:: {Ion} not found in Isotopes_dict, please Check")
            print(f"################################################################")
            return []
        if len(Ions)>1:
            addstring=f"_{'-'.join(Ions)}"
        else:
            addstring=''

        # - cicla sulle varie combinazioni
        
        for Isotopes in IsotopesList:
            # - crea nome  input file 
            #SimulationNameKey = f"{Isotopes[3]}_{CRList[0]:.0f}_{CRList[-1]:.0f}_r{float(rad[0])*100:05.0f}_lat{float(lat[0])*100:05.0f}{'_TKO' if TKO else ''}"
            SimulationNameKey = f"{Isotopes[3]}{addstring}{'_TKO' if TKO else ''}_{CRList[-1]:.0f}_{CRList[0]:.0f}_r{float(rad[0])*100:05.0f}_lat{float(lat[0])*100:05.0f}"
            InputFileName=f"Input_{SimulationNameKey}.txt"
            
            OutputFileNameList.append(f"{SimulationNameKey}")

    return OutputFileNameList, IsotopesList
## --------------------------------- leggi outputfiles e crea struttura in uscita ---------------
def LoadSimulationOutput(FileName, rig_unit):
    InputEnRig        = [] # Energy Simulated
    NRegisteredPartcle = [] # number of simulated energy per input bin
    NBins_OuterEnRig  = [] # number of bins used for the output distribution
    OuterEnRig        = [] # Bin center of output distribution
    # OuterEnRig_low    = [] # Lower Bin of output distribution <-- verifica se serve tenerlo
    EnRigDistributionAtBoundary = [] # Energy distribution at heliosphere boundary

    WARNINGLIST = []
    with open(FileName) as f:
        LineCounter=0 # contatore delle linee
        for line in f:
            #if DEBUG: print(f"reading new line: {line.strip()}")
            if line.startswith("#"):
                #if DEBUG: print("skip line")
                continue
            LineCounter +=1    # this is a good line, increase the counter
            if LineCounter==1: # the first line is the number of simulated energies
                NBins = int(line)
            else:              # the other lines follow a scheme even lines are distribution parameters, odd lines are content of distribution
                if (LineCounter % 2) ==0 : # even
                    Values=line.split() # le linee pari sono composte da 6 elementi
                    InputEnRig.append( float(Values[0])) # energia ot rigidità di input simulata (a seconda dell'unità di misura dell'uotput della simulazione)
                    # Energy(float(Values[0]), MassNumber, Z) if rig_unit else
                    if int(Values[2])!=int(Values[1]):
                        if rig_unit:
                            WARNINGLIST.append(f"WARNING: registered particle for Rigidity {float(Values[0])} ({Values[2]}) is different from injected ({Values[1]})")
                            # Energy(float(Values[0]), MassNumber, Z) if rig_unit else 
                        else:
                            WARNINGLIST.append(f"WARNING: registered particle for Energy {float(Values[0])} ({Values[2]}) is different from injected ({Values[1]})")
                    NRegisteredPartcle.append(int(Values[2]))
                    NBins_OuterEnRig.append(int(Values[3]))
                    logDeltaER = float(Values[5])
                    # np.log10(1.+Energy(np.power(10, float(Values[5]))-1, MassNumber, Z)) if rig_unit else float(Values[5])
                    # Energy(float(Values[0]), MassNumber, Z)
                    LogMinER   = float(Values[4])
                    # np.log10(InputEnRig[-1])-logDeltaER/2. if rig_unit else 
                    # OuterEnRig_low.append([pow(10.,LogMinER+itemp*logDeltaER) for itemp in range(int(Values[3]))])
                    OuterEnRig.append([(pow(10.,LogMinER+itemp*logDeltaER)+pow(10.,LogMinER+(itemp+1)*logDeltaER))/2. for itemp in range(int(Values[3]))])
                    pass
                else:                      #odd
                    Values=line.split() # le linee dispari sono composte da un numero di elementi determinato nella riga precedente
                    if len(Values)!=NBins_OuterEnRig[-1]:
                        if rig_unit:
                            WARNINGLIST.append(f"WARNING: The number of saved bins for rigidity {InputEnRig[-1]} ({len(Values)}) is different from expected ({NBins_OuterEnRig[-1]})")
                        else:
                            WARNINGLIST.append(f"WARNING: The number of saved bins for energy {InputEnRig[-1]} ({len(Values)}) is different from expected ({NBins_OuterEnRig[-1]})")
                    EnRigDistributionAtBoundary.append([ float(VV) for iV, VV in enumerate(Values) ])
                    # dlogR_dlogT(float(OuterEnRig[LineCounter//2-1][iV]), MassNumber, Z)*float(VV) if rig_unit else
                    pass
    #print("--- %s seconds ---" % (time.time() - s1))
    #--------------- Final Checks
    if NBins != len(InputEnRig):
        WARNINGLIST.append(f"WARNING: the number of readed outputs ({len(InputEnRig)}) is different from expected ({NBins})")

    #--------------- save to pythonfile
    # nota: si è scelto di mantenere i nomi e la struttura dei codici precedenti per avere la backcompatibility
    # nota2: per una gestione migliore della memoria di numpy occorre creare un "object" in modo che si crei un array di oggetti
    arr_InputEnRig                  = np.empty(NBins, object)   
    arr_NRegisteredPartcle           = np.empty(NBins, object)  
    arr_OuterEnRig                  = np.empty(NBins, object)  
    # arr_OuterEnRig_low              = np.empty(NBins, object)  
    arr_EnRigDistributionAtBoundary = np.empty(NBins, object)  
    arr_InputEnRig[:]                  =InputEnRig                                    
    arr_NRegisteredPartcle[:]           =NRegisteredPartcle                      
    arr_OuterEnRig[:]                  =OuterEnRig                                    
    # arr_OuterEnRig_low[:] =OuterEnRig_low                        
    arr_EnRigDistributionAtBoundary[:] =EnRigDistributionAtBoundary

    # Debugging
    # if len(WARNINGLIST)==0:
    #     print("NO WARNINGS")
    # else:
    #     for warn in WARNINGLIST:
    #         print(warn)

    return {
      'InputEnRig':To_np_array(arr_InputEnRig),
      'NGeneratedPartcle':To_np_array(arr_NRegisteredPartcle),
      #'OuterEnRig_low':To_np_array(arr_OuterEnRig_low),
      'OuterEnRig':To_np_array(arr_OuterEnRig),
      'BounduaryDistribution':To_np_array(arr_EnRigDistributionAtBoundary)
      }

## --------------------------------- MAIN CODE ---------------------------------------------------
if __name__ == "__main__":

    code_name = input("Type the partial path to the data to analyze (starting from Cosmica_1D, default = Cosmica_1D): ")
    if code_name=="":
        code_name = "Cosmica_1D"
    code_name = code_name + "-" + input("Type the code sub-version (if present) : ")
    rig_unit = True if "rigi" in code_name else False
    version = input("Type the setup of the code (default = 1D) : ")
    if version=="":
        version = "1D"
    sim_list_name = input("Type the simulation list name (default = Simulations_test) : ")
    if sim_list_name=="":
        sim_list_name = "Simulations_test"
    HelMod_opt = bool(input("Type 'True' if you want to analyze HelMod instead of cosmica (default = False) : "))

    SOURCECODEDIR, PASTPARAMETERLIST, FRCSTPARAMETERLIST, SIMLIST = init_sims_path(version, sim_list_name, HelMod_opt)

    # -------------------------------------------------------------------
    # carica il file dei parametri heliosferici
    HeliosphericParameters = np.loadtxt(PASTPARAMETERLIST, unpack=True)
    # la lista delle carrington rotation è decrescente dal più recente fino al passato
    FRCHeliosphericParameters = np.loadtxt(FRCSTPARAMETERLIST, unpack=True)
    HeliosphericParameters = np.append(FRCHeliosphericParameters, HeliosphericParameters, axis=1)

    ###### Crea lista dei file eseguibili processati nella cartella
    ## questi sono salvati in un file in EXELIST 
    ## da porre attenzione che potrebbero esserci doppioni
    
    ###### Carica la lista delle simulazioni che dovrebbero essere presenti nella cartella
    SimList=[]
    for line in open(SIMLIST).readlines():
        if not(line.startswith("#")):
            SingleSim=line.replace("\t","").split("|")[:8] # le colonne utili alla simulazione sono solo le prime 8, le altre contengono altri campi opzionali
            SimList.append(SingleSim) 
    #print(SimList)   

    print(f"... processing directory {code_name}")
    if version=="1D":
        ThisSimulation="%s/%s/runned_tests"%(SOURCECODEDIR, code_name)
    else:
        ThisSimulation="%s/%s"%(SOURCECODEDIR, code_name)

    ###### all'interno della cartella degli eseguibili ci sono le cartelle con le simulazioni
    ## le simulazioni sono composte poi da input files uno per ogni isotopo da simulare.
    ## l'output finale contiene tutti gli isotopi
    # - apri Simulation List

    for SimName,Ions,FileName,InitDate,FinalDate,rad,lat,lon in SimList:
        if "," in Ions:
            Ions=[s.strip() for s in Ions.split(',')]
            Ionstr='-'.join(Ions)
        else:
            Ions=[Ions.strip()]
            Ionstr=Ions[0]
        TKO= False if ("Rigidity" in FileName) else True

        InputDirPATH=f"{ThisSimulation}/{SimName.strip()}"
        OutputFile=f"{InputDirPATH}/{Ionstr}{'_TKO' if TKO  else ''}_RawMatrixFile.pkl"
        print(f"... processing Ion {Ionstr}")
        # se non esiste salta la cartella
        if not os.path.exists(InputDirPATH):
            print(f"dir {InputDirPATH} not found")
            continue
        # se esiste già output salta la cartella
        if (os.path.isfile(OutputFile)) and (not FORCE_RECALC):
            continue

        # restituisci la lista dei nomi di output attesi
        OutputFileNameList, IsotopesList = CreateOutputList(SimName, Ions, f"{SOURCECODEDIR}/DataTXT/{FileName.strip()}", InitDate, FinalDate, [rad], [lat], [lon])
        #print(OututFileNameList)
        # verifica che i file esistano
        if SINGLE_ISOTOPE:
            OutputFileNameList = [OutputFileNameList[0]]
            IsotopesList = [IsotopesList[0]]

        print(IsotopesList)

        AllFilesExist=True
        for OutputFileName in OutputFileNameList:
            ActualOutputList = glob.glob(f"{InputDirPATH}/outfile/{OutputFileName}*")
            if len(ActualOutputList) == 0:
                # se il file non esiste segnalalo
                AllFilesExist=False
                print(f"{InputDirPATH}/outfile/{OutputFileName}* : file(s) not found ")
        # se non ci sono tutti i file, skippa lo ione
        if not AllFilesExist:
            continue

        # carica i file di output e somma eventuali risultati doppi
        RawMatrixFile={}
        for iIsot  in range(len(IsotopesList)) :
            OutputFileName=OutputFileNameList[iIsot]
            Isotope = IsotopesList[iIsot][3]
            No=0 # number of outputfile for each isotopes
            IsotopeRawMatrixFile={}
            for name in glob.glob(f"{InputDirPATH}/outfile/{OutputFileName}*"):
                if name.endswith(".dat"):
                    No += 1
                    if No == 1:
                        IsotopeRawMatrixFile = LoadSimulationOutput(name, rig_unit)
                    else:
                        OtherRawMatrixFile = LoadSimulationOutput(name, rig_unit)
                        if not np.array_equal(IsotopeRawMatrixFile['InputEnRig'], OtherRawMatrixFile['InputEnRig']):
                            print(f"{name} file ignored - ERR different InputEnRig")
                            continue
                        else:
                            print(f"{name} should be included - (This is not yet implemented)")
                            # qui andrà messo la procedura che somma i risultati a parità di isotopo
            RawMatrixFile[Isotope] = IsotopeRawMatrixFile

        with open(OutputFile, "wb") as f:
            pickle.dump(RawMatrixFile, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"writed... {OutputFile}")


        #### Close and Clean

    exit()
