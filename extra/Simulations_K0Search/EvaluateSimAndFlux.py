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
import astropy.io.fits as pyfits # per elaborare i fits file
from scipy.interpolate import interp1d # interpolazione
from SimFunctions import Isotopes_dict,findKey,Energy,beta_,Rigidity,To_np_array,LinLogInterpolation,Tkin2Rigi_FluxConversione,dR_dT,dT_dR
# -- directory e nomi file generali
LISPATH_root="/home/nfsdisk/devSDT/Simulations_K0Search/LIS/"
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


## --------------------------------- leggi outputfiles e crea struttura in uscita ---------------
def LoadSimulationOutput(FileName,DEBUG=False):
    InputEnergy        = [] # Energy Simulated
    NRegisteredPartcle = [] # number of simulated enery per input bin
    NBins_OuterEnergy  = [] # number of bins used for the output distribution
    OuterEnergy        = [] # Bin center of output distribution
    # OuterEnergy_low    = [] # Lower Bin of output distribution <-- verifica se serve tenerlo
    EnergyDistributionAtBoundary = [] # Energy distribution at heliosphere boundary

    #--------------------------
    fileNameElements=FileName.split("/")[-1]
    K0Val= float(fileNameElements.split("_")[1]+"."+fileNameElements.split("_")[2])
    Isotope=fileNameElements.split("_")[0]
    if DEBUG: print(fileNameElements,":",Isotope,"-->",K0Val)
    #--------------------------

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
                    InputEnergy.append(float(Values[0])) # energia di input simulata
                    if int(Values[2])!=int(Values[1]):
                        WARNINGLIST.append(f"WARNING: registered particle for Energy {Values[0]} ({Values[2]}) is different from injected ({Values[1]})")
                    NRegisteredPartcle.append(int(Values[2]))
                    NBins_OuterEnergy.append(int(Values[3]))
                    LogMinE   = float(Values[4])
                    logDeltaE = float(Values[5])
                    # OuterEnergy_low.append([pow(10.,LogMinE+itemp*logDeltaE) for itemp in range(int(Values[3]))])
                    OuterEnergy.append([(pow(10.,LogMinE+itemp*logDeltaE)+pow(10.,LogMinE+(itemp+1)*logDeltaE))/2. for itemp in range(int(Values[3]))])
                    pass
                else:                      #odd
                    Values=line.split() # le linee dispari sono composte da un numero di elementi determinato nella riga precedente
                    if len(Values)!=NBins_OuterEnergy[-1]:
                        WARNINGLIST.append(f"WARNING: The number of saved bins for energy {InputEnergy[-1]} ({len(Values)}) is different from expected ({NBins_OuterEnergy[-1]})")
                    EnergyDistributionAtBoundary.append([ float(VV) for VV in Values ])
                    pass
    #print("--- %s seconds ---" % (time.time() - s1))
    #--------------- Final Checks
    if NBins != len(InputEnergy):
        WARNINGLIST.append(f"WARNING: the number of readed outputs ({len(InputEnergy)}) is different from expected ({NBins})")

    #--------------- save to pythonfile
    # nota: si è scelto di mantenere i nomi e la struttura dei codici precedenti per avere la backcompatibility
    # nota2: per una gestione migliore della memoria di numpy occorre creare un "object" in modo che si crei un array di oggetti
    arr_InputEnergy                  = np.empty(NBins, object)   
    arr_NRegisteredPartcle           = np.empty(NBins, object)  
    arr_OuterEnergy                  = np.empty(NBins, object)  
    # arr_OuterEnergy_low              = np.empty(NBins, object)  
    arr_EnergyDistributionAtBoundary = np.empty(NBins, object)  
    arr_InputEnergy[:]                  =InputEnergy                                    
    arr_NRegisteredPartcle[:]           =NRegisteredPartcle                      
    arr_OuterEnergy[:]                  =OuterEnergy                                    
    # arr_OuterEnergy_low[:] =OuterEnergy_low                        
    arr_EnergyDistributionAtBoundary[:] =EnergyDistributionAtBoundary  

    return {
      'K0':K0Val,
      'InputEnergy':To_np_array(arr_InputEnergy),
      'NGeneratedPartcle':To_np_array(arr_NRegisteredPartcle),
      #'OuterEnergy_low':To_np_array(arr_OuterEnergy_low),
      'OuterEnergy':To_np_array(arr_OuterEnergy),
      'BounduaryDistribution':To_np_array(arr_EnergyDistributionAtBoundary)
      },Isotope


def GetRawMatrix(InputDirPATH,OutputFileNameList,DEBUG=False):
  if not os.path.exists(InputDirPATH): 
      print(f"ERROR: dir {InputDirPATH} not found")
      exit()
  # if "," in Ions:
  #     Ions=[s.strip() for s in Ions.split(',')]
  #     Ionstr='-'.join(Ions)
  # else:
  #     Ions=[Ions.strip()]
  #     Ionstr=Ions[0]
  # carica i file di output e somma eventuali risultati doppi
  RawMatrixFile={}
  for iIsot  in range(len(OutputFileNameList)) :
      OutputFileName=OutputFileNameList[iIsot]
      #Isotope = IsotopesList[iIsot][3]
      No=0 # number of outputfile for each isotopes
      IsotopeRawMatrixFile={}
      for name in glob.glob(f"{InputDirPATH}/outfile/{OutputFileName}*"):
          No+=1
          if No==1:
              IsotopeRawMatrixFile,Isotope=LoadSimulationOutput(name,DEBUG)
          else:
              OtherRawMatrixFile,Isotope=LoadSimulationOutput(name,DEBUG)
              if not np.array_equal(IsotopeRawMatrixFile['InputEnergy'],OtherRawMatrixFile['InputEnergy']):
                  print(f"{name} file ignored - ERR different InputEnergy")
                  continue
              else:
                  print(f"{name} should be included - (This is not yet implemented)")
                  # qui andrà messo la procedura che somma i risultati a parità di isotopo
      #get k0
      K0Val=IsotopeRawMatrixFile['K0']
      if K0Val not in list(RawMatrixFile.keys()):
        RawMatrixFile[K0Val]={}

      RawMatrixFile[K0Val][Isotope]=IsotopeRawMatrixFile
  return RawMatrixFile


def EvaluateFlux(eSimList,RawMatrixFile,ExpEnRig,DEBUG=False):
  SimName  = eSimList[0].strip()
  Ion      = eSimList[1].strip()
  FileName = eSimList[2].strip()
  TKO= False if ("Rigidity" in FileName) else True
  Ion=Ion.strip()
  # - ottieni la lista degli isotopi da simulare
  IsotopesList=findKey(Ion,Isotopes_dict)          
  if DEBUG: print(IsotopesList)
  if len(IsotopesList)<=0: 
      print(f"################################################################")
      print(f"WARNING:: {Ion} not found in Isotopes_dict, please Check")
      print(f"################################################################")
      exit()
  # carica il file del LIS
  LISPATH=f"{LISPATH_root}{LISPATH_dict[Ion]}"
  if DEBUG:
    print(f"... LIS {LISPATH_dict[Ion]}")
  # carica i file di output e somma eventuali risultati doppi
  if DEBUG: 
    print(RawMatrixFile.keys())

  K0vals=list(RawMatrixFile.keys())
  Flux={}
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
    Flux[K0]=[SimEnRig,SimFlux]
    if DEBUG:
      print(f" for K0={K0} flux--> ",Flux[K0])
    # K0string=f"{K0:.6e}".replace('.',"_")
    # np.savetxt("output/ModK0Flux_%s_%s_%s_%s.dat" %(K0string,CodeVersionName,SimName,Ion),np.c_[SimEnRig,SimFlux,SimLIS,ExpFlux,Exp_Error[0],Exp_Error[1]])
  return Flux


