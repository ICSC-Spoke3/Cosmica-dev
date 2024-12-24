# -*- coding: utf-8 -*-
# developed: Dic 2023
# last update: Dic 2023
''' 
  description: 
  funzioni utili la creazione dei Job
               
'''
# -- libraries
import numpy as np
import os                                 # OS directory manager
import errno                              # error names
import shutil                             # for moving files
import datetime as dt
from SimFunctions import Isotopes_dict,findKey,Energy
#### Directory del codice sorgente
FARM='LACITY'
# SIMROOTDIR="/home/nfsdisk/devSDT/Simulations_K0Search/"
SIMROOTDIR="/Users/matteograzioso/Desktop/Research/SDEGnO/Cosmica-dev/extra/Simulations_K0Search/"
CodeVersionName="HelMod-4-CUDA_v3_1"
EXECOMMENT=f"Default settings (fast math)"
ADDITIONALMAKEFLAGS=f"""
# DA RIVEDERE per ottimizzazione tempo
VAR+=" -DMaxValueTimeStep=50 "
VAR+=" -DMinValueTimeStep=0.01 "
"""
SOURCECODEDIR=f"{SIMROOTDIR}/HM4C_Code/"
PASTPARAMETERLIST ="0_OutputFiles_v3/ParameterListALL_v12.txt"
FRCSTPARAMETERLIST="0_OutputFiles_v3/Frcst_param.txt"
TotNpartPerBin=1200 #5024
NHELIOSPHEREREGIONS=15
EXE_full_path={
    "FARMSettings":FARM,
    "CodeName":CodeVersionName,
    "SOURCECODEDIR":SOURCECODEDIR,
    "ADDITIONALMAKEFLAGS":ADDITIONALMAKEFLAGS,
}
FORCE_EXECUTE=True
def mkdir_p(path):
    # --- this function create a folder and check if this already exist ( like mkdir -p comand)
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass

def Load_HeliosphericParameters(DEBUG): 
    #-------------------------------------------------------------------
    ###### carica il file dei parametri heliosferici
    Hpar=np.loadtxt(f"{SOURCECODEDIR}{PASTPARAMETERLIST}",unpack=True) # ex HeliosphericParameters
                # la lista delle carrington rotation è decrescente dal più recente fino al passato

    FRCHeliosphericParameters=np.loadtxt(f"{SOURCECODEDIR}{FRCSTPARAMETERLIST}",unpack=True)
    Hpar=np.append(FRCHeliosphericParameters,Hpar,axis=1)
    if DEBUG:
      print(" ----- HeliosphericParameters loaded ----")
    return Hpar
    #-------------------------------------------------------------------


## ------------- Crea input Files per la simulazione ---------------
def CreateInputFile(k0vals,HeliosphericParameters,R_range,InputDirPATH,SimName,Ions,FileName,InitDate,FinalDate,rad,lat,lon,DEBUG=False):
    # - lista di uscita
    InputFileNameList=[]
    OutputrootFileNameList=[]
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

    ##############################
    #K0 scan
    # k0scanMin=5e-5 #k0RefVal-7*k0RefVal_rel*k0RefVal
    # k0scanMax=6e-4 #k0RefVal+2*k0RefVal_rel*k0RefVal
    # k0vals=np.linspace(k0scanMin, k0scanMax, num=600, endpoint=True)
    ##############################
    base_addString=addstring
    for k0val in k0vals:
      k0valstr=f"{k0val:.6e}".replace('.',"_")
      addstring=f"{base_addString}_{k0valstr}"
      if DEBUG: print(f"K0 to be simulated {k0valstr}")
      for Ion in Ions:
        # - ottieni la lista degli isotopi da simulare
        IsotopesList=findKey(Ion,Isotopes_dict)
        if DEBUG: print(IsotopesList)
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
                    Tcentr = Tcentr[(Tcentr >= R_range[0]) & (Tcentr <= R_range[1])]
                else:
                    Rigi = np.loadtxt(FileName,unpack=True,usecols=(0))
                    Rigi = Rigi[(Rigi >= R_range[0]) & (Rigi <= R_range[1])]
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
                    
                    if thisCR in CRListParam:
                        #print(thisCR,CRListParam,CRListParam.index(thisCR))
                        target.write(f"HeliosphericParameters: %.6e,\t%.3f,\t%.2f,\t%.2f,\t%.3f,\t%.3f,\t%.0f,\t%.0f,\t%.3f,\t%.2f,\t%.2f,\t%.2f,\t%.2f\n"%(
                                                            k0val if CRListParam.index(thisCR)==0 else 0. ,                          #k0
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
                OutputrootFileNameList.append(f"{SimulationNameKey}")

    return InputFileNameList,OutputrootFileNameList

## ------------- aggiungi la simulazione ai run da eseguire --------
def CreateRun(InputDirPATH,runName,EXE_full_path,InputFileNameList,DEBUG=False):
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
$EXE -i {InputFileName}  {"-vvv" if DEBUG else "-v"}
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

## CREA LE SIMULAZIONI E SOTTOMETTILE AL CLUSTER
## in uscita restituisce la lista dei file di output che vengono generati a fine simulazione
# import subprocess
# def SubmitSims(elSimList,K0arr,R_range=[],DEBUG=False):
#     Hpar=Load_HeliosphericParameters(DEBUG)
#     STARTDIR= os.getcwd()
#     SimName,Ions,FileName,InitDate,FinalDate,rad,lat,lon=elSimList
#     if "," in Ions:
#         Ions=[s.strip() for s in Ions.split(',')]
#         Ionstr='-'.join(Ions)
#     else:
#         Ions=[Ions.strip()]
#         Ionstr=Ions[0]
#     TKO= False if ("Rigidity" in FileName) else True
#     InputDirPATH=f"{STARTDIR}/{SimName}_{Ionstr}"
#     # crea la cartella della simulazione
#     # se non esiste già crea la cartella della simulazione
#     if not os.path.exists(InputDirPATH): 
#         mkdir_p(InputDirPATH)
#     # crea l'input file per la simulazione
#     #      nota che la posizione è passata come un vettore affinchè sia già pronto 
#     #      per gestire anche i casi in cui la posizione è data da un altro file
#     InputFileNameList,OutputrootFileNameList=CreateInputFile(K0arr,Hpar,R_range,InputDirPATH,SimName,Ions,f"{STARTDIR}/DataTXT/{FileName.strip()}",InitDate,FinalDate,[rad],[lat],[lon],DEBUG=DEBUG) 
#     if len(InputFileNameList)>0 : 
#         # inputfile nuovo significa che la simulazione va eseguita, inseriscila quindi nella lsita dei run
#         CreateRun(InputDirPATH,f"{SimName}_{Ionstr}{'_TKO' if TKO  else ''}",EXE_full_path,InputFileNameList,DEBUG=DEBUG)
#         # for InputFileName in InputFileNameList:
#         #     VersionLog.write(f"{InputFileName} simulated with {CodeVersionName}: {EXECOMMENT}\n")
#         #SubmitAll_arr[InputDirPATH]=1
#         # Sequenza di comandi Bash
#         bash_commands = f"cd {InputDirPATH}; condor_submit HTCondorSubFile.sub; cd {STARTDIR}"

#         # Esegui la sequenza di comandi Bash
#         result = subprocess.run(['bash', '-c', bash_commands], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

#         # Stampa l'output del comando
#         if DEBUG:
#             print("Output del comando:")
#             print(result.stdout)

#         # Stampa l'eventuale errore
#         if result.stderr:
#             print("Errore:")
#             print(result.stderr)

#     return InputDirPATH,OutputrootFileNameList

import os
import subprocess

def SubmitSims(simulation_list, k0_array, R_range=[], debug=False):
    """
    Creates simulations and executes them locally or on a cluster.
    Returns the list of output files generated after the simulation.

    Args:
        simulation_list (list): Simulation parameters including simulation name, ions, etc.
        k0_array (list): Array of K0 values to simulate.
        range_values (list, optional): Range of values for simulation. Defaults to None.
        debug (bool): If True, print debug information. Defaults to False.

    Returns:
        tuple: Path to the input directory and list of output file names.
    """
    if R_range is None:
        R_range = []
    
    # Load heliospheric parameters
    heliospheric_parameters = Load_HeliosphericParameters(debug)
    start_dir = os.getcwd()
    
    # Unpack simulation list
    sim_name, ions, file_name, init_date, final_date, radius, latitude, longitude = simulation_list
    
    # Process ions information
    if "," in ions:
        ions = [ion.strip() for ion in ions.split(',')]
        ions_str = '-'.join(ions)
    else:
        ions = [ions.strip()]
        ions_str = ions[0]
    
    # Determine the type of simulation (based on file name content)
    is_tko = "Rigidity" not in file_name
    input_dir_path = os.path.join(start_dir, f"{sim_name}_{ions_str}")
    
    # Create the simulation directory if it does not exist
    os.makedirs(input_dir_path, exist_ok=True)
    
    # Generate input and output file names for the simulation
    input_file_list, output_file_list = CreateInputFile(
        k0_array,
        heliospheric_parameters,
        R_range,
        input_dir_path,
        sim_name,
        ions,
        os.path.join(start_dir, "DataTXT", file_name.strip()),
        init_date,
        final_date,
        [radius],
        [latitude],
        [longitude]
    )
    
    if input_file_list:
        # If there are input files, prepare and run the simulation
        CreateRun(
            input_dir_path,
            f"{sim_name}_{ions_str}{'_TKO' if is_tko else ''}",
            EXE_full_path,
            input_file_list,
        )
        
        # Execute the simulation using Bash commands
        bash_commands = [
            f"cd {input_dir_path}",
            f"./run_simulations.sh",
            f"cd {start_dir}"
        ]
        try:
            # Execute the commands in a shell
            result = subprocess.run(
                "; ".join(bash_commands),
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if debug:
                print("Command output:")
                print(result.stdout)
            
            if result.stderr:
                print("Command error:")
                print(result.stderr)
        
        except subprocess.CalledProcessError as e:
            print(f"Error during simulation execution: {e}")
            if debug:
                print(e.output)
    
    return input_dir_path, output_file_list



# #### CreateSubmitAllFile
# SubmitFile = open(f"{STARTDIR}/SubmitAll.sh", "a")
# for paths in list(SubmitAll_arr.keys()):
#   SubmitFile.write(f"cd {paths} \n")
#   SubmitFile.write(f" condor_submit  HTCondorSubFile.sub \n")
#   SubmitFile.write(f"cd {STARTDIR} \n")
# SubmitFile.close()  
# #### Close and Clean    


###################################################
###################################################
###################################################
def SmoothTransition(InitialVal, FinalVal, CenterOfTransition, smoothness, x):
    # smooth transition between  InitialVal to FinalVal centered at CenterOfTransition as function of x
    # if smoothness== 0 use a sharp transition
    if smoothness==0:
        if (x>=CenterOfTransition): return FinalVal;
        else:                       return InitialVal;
    else: 
        return (InitialVal+FinalVal)/2.-(InitialVal-FinalVal)/2.*np.tanh((x-CenterOfTransition)/smoothness)



def K0Fit_ssn(p, SolarPhase, ssn):
    k0=0;
    GaussVar=0
    if(p>0.):
        if(SolarPhase==0):#/*Rising*/   
            k0=0.0002743-2.11e-6*ssn+1.486e-8*ssn*ssn-3.863e-11*ssn*ssn*ssn;   
            GaussVar=0.1122
        else:
            #/*Declining*/
            k0=0.0002787-1.66e-6*ssn+4.658e-9*ssn*ssn-6.673e-12*ssn*ssn*ssn;   
            GaussVar=0.1324
    else:
        if(SolarPhase==0):
            #/*Rising*/   
            k0=0.0003059-2.51e-6*ssn+1.284e-8*ssn*ssn-2.838e-11*ssn*ssn*ssn;   
            GaussVar=0.1097
        else:
            #/*Declining*/
            k0=0.0002876-3.715e-6*ssn+2.534e-8*ssn*ssn-5.689e-11*ssn*ssn*ssn;   
            GaussVar=0.14
    return (k0,GaussVar)


def K0Fit_NMC(NMC):
    return (np.exp(-10.83 -0.0041*NMC +4.52e-5*NMC*NMC),0.1045)


def K0CorrFactor(p,q, SolarPhase, tilt):
#   /*Authors: 2017 Stefano */
#   /* * description: Correction factor to K0 for the Kparallel. This correction is introduced 
#                     to account for the fact that K0 is evaluated with a model not including particle drift.
#                     Thus, the value need a correction once to be used in present model
#       \param p            solar polarity of HMF
#       \param q            signum of particle charge 
#       \param SolarPhase   0=rising / 1=Declining phase of solar activity cycle
#       \param tilt         Tilt angle of neutral sheet (in degree)
#   */
    K0Corr_maxv=1.5
    K0Corr_minv=1.
    K0Corr_p0_asc=18.
    K0Corr_p1_asc=40.
    K0Corr_p0_des=5.
    K0Corr_p1_des=53.
    K0Corr_maxv_neg=0.7
    K0Corr_p0_asc_neg=5.8
    K0Corr_p1_asc_neg=47.
    K0Corr_p0_des_neg=5.8
    K0Corr_p1_des_neg=58.

    if (q>0):
        if (q*p>0):
            if (SolarPhase==0): 
            # ascending
                return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_asc, K0Corr_p0_asc, tilt);
            else:
            # descending
                return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_des, K0Corr_p0_des, tilt); 
        else:
            return 1
    if (q<0):
        if (q*p>0):
            if (SolarPhase==0): 
            # ascending
                return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_asc, K0Corr_p0_asc, tilt);
            else:
            # descending
                return SmoothTransition(K0Corr_maxv, K0Corr_minv, K0Corr_p1_des, K0Corr_p0_des, tilt);
        else:
            if (SolarPhase==0): 
            # ascending
                return SmoothTransition(K0Corr_maxv_neg, K0Corr_minv, K0Corr_p1_asc_neg, K0Corr_p0_asc_neg, tilt);
            else: 
            # descending
                return SmoothTransition(K0Corr_maxv_neg, K0Corr_minv, K0Corr_p1_des_neg, K0Corr_p0_des_neg, tilt);  
    return 1;



def EvalK0(IsHighActivityPeriod, p, q, SolarPhase, tilt, NMC, ssn):
#   /*Authors: 2022 Stefano */
#   /* * description: Evaluate diffusion parameter from fitting procedures.
#       \param p            solar polarity of HMF
#       \param q            signum of particle charge 
#       \param SolarPhase   0=rising / 1=Declining phase of solar activity cycle
#       \param tilt         Tilt angle of neutral sheet (in degree)
#       \return x = k0_paral
#               y = k0_perp
#               z = GaussVar
#   */
#   float3 output;
    K0cor=K0CorrFactor(p,q,SolarPhase,tilt)#; // k0_paral is corrected by a correction factor   
    if (IsHighActivityPeriod and NMC>0):
        K0,Kerr=K0Fit_NMC(NMC);
    else:
        K0,Kerr=K0Fit_ssn(p, SolarPhase, ssn)      
    return (K0*K0cor,Kerr);




### genera la griglia dei K0
def griglia_valori_k0(time="",NK0=1,Hpar=[],q=+1,DEBUG=False):
    time=int(time)
    K0grid = np.ones(NK0)
    K0,K0err = -1,1
    # carica il file dei parametri e cerca i parametri di riferimento
    for iparam,line in enumerate(Hpar[0]):
        #print(iparam,line)
        begCR = int(Hpar[0][iparam])
        endCR = int(Hpar[1][iparam])
        #print(f"{time} {begCR} {endCR}")
        if time>=begCR and time<endCR:
            ssn         = Hpar[2][iparam]
            #V0          = Hpar[3][iparam]
            Tilt_L      = Hpar[4][iparam]
            SmoothTiltL = Hpar[12][iparam]
            #Bfield      = Hpar[6][iparam]
            Polarity    = Hpar[7][iparam]
            SolarPhase  = Hpar[8][iparam]
            NMCR        = Hpar[11][iparam]
            IsHighActivityPeriod = (np.average([ float(tilt) for tilt in Hpar[4][iparam:iparam+15] ]))>50
            K0,K0err=EvalK0(IsHighActivityPeriod, Polarity, q, SolarPhase, Tilt_L, NMCR, ssn)
            if DEBUG:
                print(f"IsHighActivityPeriod {IsHighActivityPeriod}")
                print(f"K0 {K0} +- {K0err*K0}")
            #griglia uniforme 
            K0grid=np.linspace(K0-4*K0err*K0,K0+4*K0err*K0,num=NK0,endpoint=True)
            break
    return K0grid,K0,K0err
