# -*- coding: utf-8 -*-
# developed: Aug 2023
# last update: Sep 2023
''' 
  description: procedura di ricerca del K0
               
'''
# -- libraries
import numpy as np
import os                                 # OS directory manager
import errno                              # error names
import shutil                             # for moving files
import datetime as dt
import glob
import time as timeobj                    # for sleep 
import shutil                             # for rmtree
import pickle
#from SimFunctions import *
from JobCreation import Load_HeliosphericParameters
from JobCreation import griglia_valori_k0
from JobCreation import SubmitSims
from EvaluateSimAndFlux import GetRawMatrix
from EvaluateSimAndFlux import EvaluateFlux
DEBUG=False
if DEBUG:
  import json
NK0=400
SleepStime=60
#STARTDIR = os.getcwd()
#---------------------------------------------
#------ Variabile di output
Output_dict={}
App="_FD006"
#--------------------------------------------
#------ carica lista dei giorni da simulare
SimList=[]
for line in open(f"Simulations{App}.list").readlines():
    if not(line.startswith("#")):
        SingleSim=line.replace("\t","").split("|")[:8] # le colonne utili alla simulazione sono solo le prime 8, le altre contengono altri campi opzionali
        SingleSim=[x.strip() for x in SingleSim]
        ###################################################
        ## qui mettere eventuali condizioni di taglio sulla lista
        ## ad esempio la condizione:
        ##            float(SingleSim[3])>=20230601 and float(SingleSim[3])< 20231015
        ##            esclude i periodi dentro l'intervallo
        if DEBUG:
          # intervallo per il DEBUG del codice
          if float(SingleSim[3])<20180101 or float(SingleSim[3])> 20180102:
            continue
        if "Electron" in SingleSim[1] or "Positron" in SingleSim[1]:
          if DEBUG: 
            print(f"Excluding {SingleSim} from simulations to be done")
          continue
        ###################################################
        SimList.append(SingleSim) 
        Ions=SingleSim[1].strip()
        if Ions not in Output_dict:
          Output_dict[Ions]={}
if DEBUG:
  print(f" ----- Simulation list loaded ----") 
  for SingleSim in SimList:
    print(SingleSim)




Hpar=Load_HeliosphericParameters(DEBUG)

#---------------------------------------------
#------ ciclo di ricerca
# for each day of sim 
for elSimList in SimList:
  #if DEBUG:
  print(f"= = = {elSimList[0]} = = = {elSimList[1]} = = = ")
  ThisIon=elSimList[1].strip()
  # elementi in elSimList
  #SimName,Ions,FileName,InitDate,FinalDate,rad,lat,lon
  time=dt.datetime.strptime(elSimList[3], '%Y%m%d').date()

  #.. Crea griglia valori di K0 da simulare
  K0arr,K0ref,K0ref_err= griglia_valori_k0(time=elSimList[3],NK0=NK0,Hpar=Hpar,DEBUG=DEBUG)
  if DEBUG:
    print(f"--> Simuleremo {NK0} K0 : {K0arr}")
    print(f"--> K0ref {K0ref} K0ref_err {K0ref_err}")
  #.. leggi e salva il dato sperimentale
  FileName="DataTXT/"+elSimList[2]
  xExp,fExp,ErExp_inf,ErExp_sup = np.loadtxt(FileName,unpack=True,usecols=(0,1,2,3))
  ErExp_sup=ErExp_sup[(xExp >= 3) & (xExp <= 11)]
  ErExp_inf=ErExp_inf[(xExp >= 3) & (xExp <= 11)]
  fExp=fExp[(xExp >= 3) & (xExp <= 11)]
  xExp=xExp[(xExp >= 3) & (xExp <= 11)] 
  #.. inizializza dizionario output 
  for iT,T in enumerate(xExp):
    if not (T>=3 and T<=11): # -- seleziona solo le energie interessanti per questo studio
      continue
    if T not in Output_dict[ThisIon]:
      Output_dict[ThisIon][T]={}
    Output_dict[ThisIon][T][time]={
                      "diffBest"  : 9e99, # migliore differenza Dati-Simulazione
                      "K0best"    : 0,
                      "K0Min"     : 9e99,
                      "K0Max"     : 0,
                      "Fluxbest"  : 0,
                      "FluxMin"   : 9e99,
                      "FluxMax"   : 0,
                      #"xExp"      : xExp[iT],
                      "fExp"      : fExp[iT],
                      "ErExp_inf" : ErExp_inf[iT],
                      "ErExp_sup" : ErExp_sup[iT],
                      "K0ref"     : K0ref,
                      "K0Err_ref" : K0ref_err*K0ref,
                      }
  # if DEBUG:
  #   for T in list(Output_dict['Proton'])[:1]:
  #     for time in Output_dict['Proton'][T]:
  #       print(f"-- {T} GV {time}------------------")
  #       print(json.dumps(Output_dict['Proton'][T][time], indent=4))


  #.. crea le simulazioni per ogni K0 e sottomettile tutte in blocco
  #   tieni però traccia dei file di output da attendere
  OutputDirPATH,OutputFileList = SubmitSims(elSimList,K0arr,R_range=[3,11],DEBUG=DEBUG)

  #TODO: riscrivere fin qui FINO A CreateInputFile dentro a SubmitSims

  #.. Attenti che tutti gli output siano finiti
  #   conta il numero di file generati, quando il numero è pari a quello atteso, passa alla nuova fase
  NFileAttesi = len(OutputFileList) 
  OutputDirPATHName=f"{OutputDirPATH}/outfile/*.dat"
  print(f"waiting...  {NFileAttesi} files expected")
  while len(glob.glob(OutputDirPATHName))<NFileAttesi:
    timeobj.sleep(SleepStime)
    if DEBUG:
      print(f"Found {len(glob.glob(OutputDirPATHName))} files on {NFileAttesi} expected")
  #.. Con tutti output completi, carica la RawMatrix
  #   questo emula la creazione del file pkl
  RawMatrixSims=GetRawMatrix(OutputDirPATH,OutputFileList,DEBUG=DEBUG)

  #.. calcola il Flusso di questo blocco di simulazioni
  Fluxes =  EvaluateFlux(elSimList,RawMatrixSims,xExp,DEBUG=DEBUG)

  #.. cicla sulle soluzioni alla ricerca del best value
  for K0, valore in Fluxes.items():
    SimEnRig,SimFlux=valore
    if DEBUG: print(SimEnRig,Output_dict[ThisIon].keys())
    for iF,F in enumerate(SimFlux):
      T=xExp[iF]
      RefEntry=Output_dict[ThisIon][T][time]
      DiffVal = F-RefEntry['fExp']
      if -RefEntry['ErExp_inf']<=DiffVal<=+RefEntry['ErExp_sup'] :
        if np.fabs(DiffVal)<np.fabs(RefEntry['diffBest']):
          Output_dict[ThisIon][T][time]['diffBest']=DiffVal
          Output_dict[ThisIon][T][time]['K0best']  =K0
          Output_dict[ThisIon][T][time]['Fluxbest']=F
        if RefEntry['K0Min']>K0 :
          Output_dict[ThisIon][T][time]['FluxMin']=F
          Output_dict[ThisIon][T][time]['K0Min']  =K0
        if RefEntry['K0Max']<K0 :
          Output_dict[ThisIon][T][time]['FluxMax']=F
          Output_dict[ThisIon][T][time]['K0Max']  =K0
  #.. cancella tutto

  if OutputDirPATH!='/' and OutputDirPATH!='/home/' and OutputDirPATH!='/home/nfdisk/' and 'Simulations_K0Search' in OutputDirPATH:
    #if DEBUG:
    print(f"Cancellando la cartella {OutputDirPATH}")
    shutil.rmtree(OutputDirPATH)
  #    
  # Salva output Dizionario
  with open(f"Result{App}.pkl", "wb") as f:
    pickle.dump(Output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
  print(f"writed... Result{App}.pkl")







