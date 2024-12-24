import numpy as np
from utils import To_np_array
"""
Simulation_Output_handler.py
This module provides functionality to load and process simulation output files. It reads the simulation data from a specified file and returns it in a structured format along with any warnings encountered during the reading process.
Functions:
    LoadSimulationOutput(FileName)
        Load a simulation output file and return the data in a dictionary along with a list of warnings.
Usage Example:
    SimulationOutput, Warnings = LoadSimulationOutput(FileName)
    for iEnergy, Energy in enumerate(SimulationOutput['InputEnergy']):
        plot_ascii(SimulationOutput['OuterEnergy'][iEnergy], SimulationOutput['BounduaryDistribution'][iEnergy])
    if len(Warnings) > 0:
"""


def LoadSimulationOutput(FileName):
    """Load a simulation output file
    FileName: string, path to the file
    Returns: dictionary, containing the simulation output
        - InputEnergy: numpy array, containing the input energies
        - NGeneratedPartcle: numpy array, containing the number of generated particles
        - OuterEnergy: numpy array, containing the outer energies
        - BounduaryDistribution: numpy array, containing the boundary distribution
        and a list of warnings
    """
    InputEnergy        = [] # Energy Simulated
    NRegisteredPartcle = [] # number of simulated enery per input bin
    NBins_OuterEnergy  = [] # number of bins used for the output distribution
    OuterEnergy        = [] # Bin center of output distribution
    # OuterEnergy_low    = [] # Lower Bin of output distribution <-- verifica se serve tenerlo
    EnergyDistributionAtBoundary = [] # Energy distribution at heliosphere boundary
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
            'InputEnergy':To_np_array(arr_InputEnergy),
            'NGeneratedPartcle':To_np_array(arr_NRegisteredPartcle),
            #'OuterEnergy_low':To_np_array(arr_OuterEnergy_low),
            'OuterEnergy':To_np_array(arr_OuterEnergy),
            'BounduaryDistribution':To_np_array(arr_EnergyDistributionAtBoundary)
            },WARNINGLIST




if __name__ == "__main__":
    from utils import plot_ascii
    FileName = "Librerie/esempi/Deuteron_Simulation_Output_Example.dat"
    SimulationOutput,Warnings = LoadSimulationOutput(FileName)
    for iEnergy,Energy in enumerate(SimulationOutput['InputEnergy']):
        print("-------------------------------------------------")
        plot_ascii(SimulationOutput['OuterEnergy'][iEnergy],SimulationOutput['BounduaryDistribution'][iEnergy])
        print(f"Energy: {Energy} GeV")
        print(f"Number of generated particles: {SimulationOutput['NGeneratedPartcle'][iEnergy]}")
        print(f"Number of bins in the output distribution: {len(SimulationOutput['OuterEnergy'][iEnergy])} [{min(SimulationOutput['OuterEnergy'][iEnergy]):.3f} - {max(SimulationOutput['OuterEnergy'][iEnergy]):.3f} GeV/n]")
        
    if len(Warnings)>0:
        print("Warnings:")    
        print(Warnings)
    else:
        print("No warnings")