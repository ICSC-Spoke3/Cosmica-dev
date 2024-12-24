# -*- coding: utf-8 -*-
# developed: Aug 2023
# last update: Aug 2023
''' 
  description: procedura di creazione dei file di input e dei run per la elaborazione numerica
               
'''
# -- libraries
import numpy as np
import os                                 # OS directory manager
import errno                              # error names
import shutil                             # for moving files
import datetime as dt
#### Directory del codice sorgente
FARM='LACITY'
SIMROOTDIR="/home/nfsdisk/"
CodeVersionName="HelMod-4-CUDA_v3"
EXECOMMENT=f"Default settings (fast math)"
ADDITIONALMAKEFLAGS=f"""
# DA RIVEDERE per ottimizzazione tempo
VAR+=" -DMaxValueTimeStep=50 "
VAR+=" -DMinValueTimeStep=0.01 "
"""
SOURCECODEDIR=f"{SIMROOTDIR}HelMod4CSims/HM4C_Code/"
PASTPARAMETERLIST ="0_OutputFiles_v3/ParameterListALL_v12.txt"
FRCSTPARAMETERLIST="0_OutputFiles_v3/Frcst_param.txt"
TotNpartPerBin=5024
NHELIOSPHEREREGIONS=15
DEBUG=True

FORCE_EXECUTE=True # quando True tutte le simulazioni create vengono messe nella lista dei run da eseguire

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
                "subIron":   [( 21., 45., 0.931494061 ,"subIron-Sc45"),
                              ( 22., 48., 0.931494061 ,"subIron-Ti48"),
                              ( 22., 44., 0.931494061 ,"subIron-Ti44"),
                              ( 22., 46., 0.931494061 ,"subIron-Ti46"),
                              ( 22., 47., 0.931494061 ,"subIron-Ti47"),
                              ( 22., 49., 0.931494061 ,"subIron-Ti49"),
                              ( 22., 50., 0.931494061 ,"subIron-Ti50"),
                              ( 23., 51., 0.931494061 ,"subIron-V51" ),
                              ( 23., 49., 0.931494061 ,"subIron-V49" ),
                              ( 23., 50., 0.931494061 ,"subIron-V50" ),
                              ( 24., 52., 0.931494061 ,"subIron-Cr52"),
                              ( 24., 50., 0.931494061 ,"subIron-Cr50"),
                              ( 24., 51., 0.931494061 ,"subIron-Cr51"),
                              ( 24., 53., 0.931494061 ,"subIron-Cr53"),
                              ( 24., 54., 0.931494061 ,"subIron-Cr54")],
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
# -- General functions
def mkdir_p(path):
    # --- this function create a folder and check if this already exist ( like mkdir -p comand)
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def Rigidity(T,MassNumber=1.,Z=1.):
  import numpy
  T0=0.931494061
  if numpy.fabs(Z)==1:
    T0 = 0.938272046
  if MassNumber==0:
          T0 = 5.11e-4
          MassNumber = 1
  return MassNumber/numpy.fabs(Z)*numpy.sqrt(T*(T+2.*T0))

def Energy(R,MassNumber=1.,Z=1.):
  import numpy
  T0=0.931494061
  if numpy.fabs(Z)==1:
    T0 = 0.938272046
  if MassNumber==0:
          T0 = 5.11e-4
          MassNumber = 1
  return numpy.sqrt((Z*Z)/(MassNumber*MassNumber)*(R*R)+(T0*T0))-T0;

def beta_(T,T0_Glob=0.938):
    import math
    T0=T0_Glob#rest mass
    tt = T + T0
    t2 = tt + T0
    beta = math.sqrt(T*t2)/tt
    return beta



## ------------- Crea input Files per la simulazione ---------------
def CreateInputFile(InputDirPATH,SimName,Ions,FileName,InitDate,FinalDate,rad,lat,lon):
    # - lista di uscita
    InputFileNameList=[]
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
        print(f"WARNING:: CR list empty {InitDate} {FinalDate}")
        return []

    # - controlla se i vettori posizioni in ingresso sono lunghi quanto CRList, 
    #   nel caso non lo fossero significa che la posizione è la stessa per tutte le simulazioni
    if len(rad)!=len(CRList):
        if len(rad)!=1:
            print("ERROR: Source array dimension != CR list selected")
            exit(1)
        nprad = np.ones(len(CRList))*float(rad[0])
        nplat = np.ones(len(CRList))*(90.-float(lat[0]))/180.*np.pi
        nplon = np.ones(len(CRList))*(float(lon[0])/180.*np.pi)
    else:
        nprad = np.array([float(x)                  for x in rad])
        nplat = np.array([(90.-float(x))/180.*np.pi for x in lat])
        nplon = np.array([float(x)/180.*np.pi       for x in lon])

    if len(Ions)>1:
        addstring=f"_{'-'.join(Ions)}"
    else:
        addstring=''

    for Ion in Ions:
        # - ottieni la lista degli isotopi da simulare
        IsotopesList=findKey(Ion,Isotopes_dict)
        print(IsotopesList)
        if len(IsotopesList)<=0: 
            print(f"################################################################")
            print(f"WARNING:: {Ion} not found in Isotopes_dict, please Check")
            print(f"################################################################")
            return []
        # - cicla sulle varie combinazioni
        
        for Isotopes in IsotopesList:
            # - crea nome  input file 
            SimulationNameKey = f"{Isotopes[3]}{addstring}{'_TKO' if TKO else ''}_{CRList[-1]:.0f}_{CRList[0]:.0f}_r{float(rad[0])*100:05.0f}_lat{float(lat[0])*100:05.0f}"

            InputFileName=f"Input_{SimulationNameKey}.txt"
            # - verifica se il file esiste, se non esiste or FORCE_EXECUTE=True allora crea il file e appendilo alla lista in InputFileNameList
            if not os.path.isfile(f"{InputDirPATH}/{InputFileName}") or FORCE_EXECUTE:
                target = open(f"{InputDirPATH}/{InputFileName}","w")
                target.write(f"#File generated on {dt.date.today()}\n")

                ############################ OutputFilename
                target.write(f"OutputFilename: {SimulationNameKey}\n")
                ############################ particle to be simulated
                target.write(f"# particle to be simulated\n")
                target.write(f"Particle_NucleonRestMass: {Isotopes[2]}\n")
                target.write(f"Particle_MassNumber: {Isotopes[1]}\n")
                target.write(f"Particle_Charge: {Isotopes[0]}\n")

                ############################ Load Energy Bins
                target.write(f"# .. Generation energies -- NOTE the row cannot exceed 2000 chars\n");  
                if  TKO:
                    Tcentr = np.loadtxt(FileName,unpack=True,usecols=(0))
                else:
                    Rigi = np.loadtxt(FileName,unpack=True,usecols=(0))
                    Tcentr=Energy(Rigi,MassNumber=Isotopes[1],Z=Isotopes[0])
                if Tcentr.size>1: 
                    Tcentr=','.join(f"{x:.3e}" for x in Tcentr)
                Tcentr=f"Tcentr: {Tcentr} \n"
                if len(Tcentr)>=2000:
                    print("ERROR: too much Energy inputs (exceeding allowed 2000 characters)")
                    exit(1)
                target.write(Tcentr);

                ############################ Source (detector) position
                target.write(f"# .. Source Position {len(nplat)}\n");
                strSourceThe=f"SourcePos_theta: {','.join(f'{x:.5f}' for x in nplat)} \n"
                strSourcePhi=f"SourcePos_phi: {','.join(f'{x:.5f}' for x in nplon)} \n"
                strSourceRad=f"SourcePos_r: {','.join(f'{x:.5f}' for x in nprad)}  \n"
                if len(strSourceThe)>=2000 or len(strSourcePhi)>=2000 or len(strSourceRad)>=2000:
                    print("ERROR: too much source points (exceeding allowed 2000 characters)")
                    exit(1)
                target.write(strSourceThe);
                target.write(strSourcePhi);
                target.write(strSourceRad);

                ############################ Source (detector) position
                target.write(f"# .. Number of particle to be generated\n");
                target.write(f"Npart: {TotNpartPerBin}\n");
                ############################ Heliosphere Parameters
                target.write(f"# .. Heliosphere Parameters\n");
                target.write(f"Nregions: {NHELIOSPHEREREGIONS}\n");
                target.write(f"# from {CRList[0]} to {CRList[-1]} - Total {len(CRListParam)}({len(CRListParam)-NHELIOSPHEREREGIONS+1}+{NHELIOSPHEREREGIONS-1}) input parameters periods\n");
                target.write(f"# . region 0 :            k0,    ssn,      V0, TiltAngle,SmoothTilt, Bfield, Polarity, SolarPhase, NMCR, Rts_nose, Rts_tail, Rhp_nose, Rhp_tail\n")
                for iparam in range(len(HeliosphericParameters[0])):
                    thisCR = HeliosphericParameters[0][iparam]
                    #print(thisCR)
                    if thisCR in CRListParam:
                        target.write(f"HeliosphericParameters: %.1f,\t%.3f,\t%.2f,\t%.2f,\t%.3f,\t%.3f,\t%.0f,\t%.0f,\t%.3f,\t%.2f,\t%.2f,\t%.2f,\t%.2f\n"%(
                                                                0. ,                          #k0
                                                            HeliosphericParameters[2][iparam], #ssn
                                                            HeliosphericParameters[3][iparam], #V0
                                                            HeliosphericParameters[4][iparam], #Tilt L
                                                            HeliosphericParameters[12][iparam], #SmoothTilt L
                                                            HeliosphericParameters[6][iparam], #Bfield
                                                            HeliosphericParameters[7][iparam], #Polarity
                                                            HeliosphericParameters[8][iparam], #SolarPhase
                                                            HeliosphericParameters[11][iparam], #NMCR
                                                            HeliosphericParameters[13][iparam], #Rts_nose
                                                            HeliosphericParameters[14][iparam], #Rts_tail
                                                            HeliosphericParameters[15][iparam], #Rhp_nose
                                                            HeliosphericParameters[16][iparam])) #Rhp_tail
                target.write(f"# . heliosheat        k0,     V0,\n");
                for iparam in range(len(HeliosphericParameters[0])):
                    thisCR = HeliosphericParameters[0][iparam]
                    #print(thisCR)
                    if thisCR in CRList:
                        target.write(f"HeliosheatParameters: %.5e,\t%.2f \n"%(3.e-05,
                                                                             HeliosphericParameters[3][iparam+NHELIOSPHEREREGIONS-1]));

                target.close()
                InputFileNameList.append(f"{InputDirPATH}/{InputFileName}")

    return InputFileNameList

## ------------- aggiungi la simulazione ai run da eseguire --------
def CreateRun(InputDirPATH,runName,EXE_full_path,InputFileNameList):
    # lo scopo di questa funzione è di creare il file dei run da eseguire
    mkdir_p(f"{InputDirPATH}/run")
    mkdir_p(f"{InputDirPATH}/outfile")
    
    # --- caso Farm Locale
    if FARM=='LACITY':
        # per ogni eseguibile crea un run 
        # vedi https://batchdocs.web.cern.ch/tutorial/exercise10.html
        for InputFileName in InputFileNameList:
            run_name=os.path.basename(InputFileName).split('.')[0] 
            ####XXXXX qui estrarre il nome dell'inputfile che diventerà il nome deljob
            ### per ogni job è del tipo qui sotto con integrazione CUDA (da scrivere)
            runsFile = open(f"{InputDirPATH}/run/HTCrun_{run_name}.run", "w")
            runsFile.write(f"""#!/bin/bash
set -e
source /etc/bashrc
echo `hostname`
whoami
HERE=$PWD
mkdir -p {InputDirPATH}/outfile
mkdir -p {InputDirPATH}/exefile

# compile exefile for actual run
COMPILER="nvcc " 
EXE="{InputDirPATH}/exefile/Execute_{EXE_full_path['CodeName']}_{run_name}"
SOURCE=" {EXE_full_path['SOURCECODEDIR']}/{EXE_full_path['CodeName']}/HelMod-4.cu  "
SOURCE+=` ls {EXE_full_path['SOURCECODEDIR']}/{EXE_full_path['CodeName']}/sources/*.cu`
HEADER="{EXE_full_path['SOURCECODEDIR']}/{EXE_full_path['CodeName']}/headers/"
# compiling options
ptxas=" --ptxas-options=\"-v\" "  # mostra dettagli registri
el=" -rdc=true "             # Enable or disable the generation of relocatable device code. If disabled, executable device code is generated. Relocatable device code must be linked before it can be executed.
                              # --> risolve il l'errore ptxas fatal   : Unresolved extern function
openmp=" -Xcompiler -fopenmp " # per usare OpenMP per il multithread
opt=" --use_fast_math "

#Optimization variables
VAR=" -DSetWarpPerBlock=10" 
#additional flags
{EXE_full_path['ADDITIONALMAKEFLAGS']}
# compiling string
$COMPILER $ptxas $el $VAR $openmp $opt  -I $HEADER -o $EXE $SOURCE

##########################
##########################


cd {InputDirPATH}/outfile
$EXE -i {InputFileName} {"-vvv" if DEBUG else ""}
#rm $EXE
echo Done....
         """)
            runsFile.close()
        if not os.path.isfile(f"{InputDirPATH}/HTCondorSubFile.sub"):
            HTcondrun = open(f"{InputDirPATH}/HTCondorSubFile.sub",'w')
            HTcondrun.write(f"""
executable              = $(filename)
arguments               = $(ClusterId)$(ProcId)
output                  = $(filename).$(ClusterId).$(ProcId).out
error                   = $(filename).$(ClusterId).$(ProcId).err
log                     = $(ClusterId).log
request_GPUs = 1
request_CPUs = 1
queue filename matching files run/*.run
                                """)
            HTcondrun.close()
        pass 
    
    return 0

## --------------------------------- MAIN CODE ---------------------------------------------------
if __name__ == "__main__":
    STARTDIR = os.getcwd()
    ## ogni job compila anche il proprio eseguibile che sarà dentro la cartella del job stesso
    ## le informazioni dell'eseguibile da compilare sono passati via parametri esterni o definiti di default
    #### ----- da implementare ... load parametri esterni
    EXE_full_path={
    "FARMSettings":FARM,
    "CodeName":CodeVersionName,
    "SOURCECODEDIR":SOURCECODEDIR,
    "ADDITIONALMAKEFLAGS":ADDITIONALMAKEFLAGS,
    }
    ###### DEBUG LINES ####
    if DEBUG:
      print(f" ----- Script initialized with values ----")
      print(f"FARM Settings      : {FARM}")
      print(f"SIMROOTDIR         = {SIMROOTDIR}")
      print(f"CodeVersionName    = {CodeVersionName}")
      print(f"EXECOMMENT         = {EXECOMMENT}")
      print(f"ADDITIONALMAKEFLAGS= {ADDITIONALMAKEFLAGS}")
      print(f"SOURCECODEDIR      = {SOURCECODEDIR}")
      print(f"PASTPARAMETERLIST  = {PASTPARAMETERLIST}")
      print(f"FRCSTPARAMETERLIST = {FRCSTPARAMETERLIST}")
      print(f"TotNpartPerBin     = {TotNpartPerBin}")    
      print(f"NHELIOSPHEREREGIONS= {NHELIOSPHEREREGIONS}")  
      print(f" ----------------------------------------")      
    ###### END DEBUG LINES ####


    #-------------------------------------------------------------------
    ###### Salva il nome dell'eseguibile in un file. 
    ##     e crea una cartella con il nome del programma.
    ##     questa cosa di salvare un logfile è fatta per semplicità di gestione e backcompatibilità. 
    ##     alla lunga verificare che in effetti serva a qualcosa
    
    if not os.path.isdir("%s"%(CodeVersionName)):
        # controlla se l'eseguibile con quel nome è già stato processato
        # !!!si suppone che nomi diversi corrisponsano parametri di compilazione differenti!!!
        # nel caso non lo fosse, crea una cartella con quel nome e appendi la simulazione alla lista
        
        EXEDIR="exefiles"
        mkdir_p(EXEDIR)
        file1 = open(f"{STARTDIR}/{EXEDIR}/CodeVersions.txt", "a")  # append mode
        file1.write(f"{CodeVersionName} |{EXECOMMENT}\n")
        file1.close()
        if DEBUG: print(f" ----- {CodeVersionName} saved in CodeVersions.txt ----")
        mkdir_p(CodeVersionName) 
        if DEBUG: print(f" ----- {CodeVersionName} saved in CodeVersions.txt ----")
    else: 
        if DEBUG: print(f" ----- {CodeVersionName} already in CodeVersions.txt ----")
    #-------------------------------------------------------------------
    
    #-------------------------------------------------------------------
    ###### carica il file dei parametri heliosferici
    HeliosphericParameters=np.loadtxt(f"{SOURCECODEDIR}/{PASTPARAMETERLIST}",unpack=True)
                # la lista delle carrington rotation è decrescente dal più recente fino al passato

    FRCHeliosphericParameters=np.loadtxt(f"{SOURCECODEDIR}/{FRCSTPARAMETERLIST}",unpack=True)
    HeliosphericParameters=np.append(FRCHeliosphericParameters,HeliosphericParameters,axis=1)
    if DEBUG:
      print(" ----- HeliosphericParameters loaded ----")
    #-------------------------------------------------------------------


    #-------------------------------------------------------------------
    os.chdir(STARTDIR) # assicurati di essere nella cartella iniziale
    ###### crea la lista delle simulazioni 
    ## La lista delle simulazioni è fornita attraverso un file dal nome "Simulations.list"
    ## presente nella stessa cartella. 
    SimList=[]
    for line in open("Simulations.list").readlines():
        if not(line.startswith("#")):
            SingleSim=line.replace("\t","").split("|")[:8] # le colonne utili alla simulazione sono solo le prime 8, le altre contengono altri campi opzionali
            SingleSim=[x.strip() for x in SingleSim]
            ###################################################
            ## qui mettere eventuali condizioni di taglio sulla lista
            ## ad esempio la condizione:
            ##            float(SingleSim[3])>=20230601 and float(SingleSim[3])< 20231015
            ##            esclude i periodi dentro l'intervallo
            if "Electron" in SingleSim[1] or "Positron" in SingleSim[1] :
              if DEBUG: 
                print(f"Excluding {SingleSim} from simulations to be done")
              continue
            ###################################################
            SimList.append(SingleSim) 
    if DEBUG:
      print(f" ----- Simulation list loaded ----") 
      for SingleSim in SimList:
        print(SingleSim)
    #-------------------------------------------------------------------

    #-------------------------------------------------------------------
    ###### Crea le cartelle delle singole simulazioni.
    ##     Le simulazioni sono composte poi da input files uno per ogni isotopo da simulare.
    ##     Mantieni un log file di quanto creato.
    ThisSimulation="%s/%s"%(STARTDIR,CodeVersionName)
    VersionLog = open(f"{ThisSimulation}/Version.txt", "a")
    SubmitAll_arr={}
    for SimName,Ions,FileName,InitDate,FinalDate,rad,lat,lon in SimList:
        if "," in Ions:
            Ions=[s.strip() for s in Ions.split(',')]
            Ionstr='-'.join(Ions)
        else:
            Ions=[Ions.strip()]
            Ionstr=Ions[0]
        TKO= False if ("Rigidity" in FileName) else True
        InputDirPATH=f"{ThisSimulation}/{SimName}"
        # se non esiste già crea la cartella della simulazione
        if not os.path.exists(InputDirPATH): 
            mkdir_p(InputDirPATH)
        # crea l'input file per la simulazione, se l'input file esisteva già la funzione non lo inserisce nella lista 
        #      altrimenti significa che è una simulazione nuova che va eseguita
        #      nota che la posizione è passata come un vettore affinchè sia già pronto per gestire anche i casi in cui la posizione è data da un altro file
        InputFileNameList=CreateInputFile(InputDirPATH,SimName,Ions,f"{STARTDIR}/DataTXT/{FileName.strip()}",InitDate,FinalDate,[rad],[lat],[lon]) 
        if len(InputFileNameList)>0 : 
            # inputfile nuovo significa che la simulazione va eseguita, inseriscila quindi nella lsita dei run
            CreateRun(InputDirPATH,f"{SimName}_{Ionstr}{'_TKO' if TKO  else ''}",EXE_full_path,InputFileNameList)
            for InputFileName in InputFileNameList:
                VersionLog.write(f"{InputFileName} simulated with {CodeVersionName}: {EXECOMMENT}\n")
            SubmitAll_arr[InputDirPATH]=1

    #### CreateSubmitAllFile
    SubmitFile = open(f"{STARTDIR}/SubmitAll.sh", "a")
    for paths in list(SubmitAll_arr.keys()):
      SubmitFile.write(f"cd {paths} \n")
      SubmitFile.write(f" condor_submit  HTCondorSubFile.sub \n")
      SubmitFile.write(f"cd {STARTDIR} \n")
    SubmitFile.close()  
    #### Close and Clean
    VersionLog.close()  
    exit()

