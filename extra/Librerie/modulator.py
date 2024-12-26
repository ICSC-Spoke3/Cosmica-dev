import sys
import numpy as np
from utils import LinLogInterpolation
from utils import beta_
from utils import To_np_array
from utils import Tkin2Rigi_FluxConversione
from Particle_Library import Isotopes_dict,findKey
from LIS import LoadLIS,GetLIS




def Spectra_BackwardEnergy(Modulation_Matrix_dict_Isotope,LIS_isotope,T0):
    """ Evaluate the modulated specra for a single isotope in case of SDE Monte Carlo in Rigidity
        Modulation_Matrix_dict_Isotope: dictionary, containing the matrix of the simulation for a specific isotope
        LIS_isotope: tuple, containing the energy bins and the LIS_isotope spectra for selected ion
        T0: float, kinetic energy of the particle at rest
        Returns: tuple, containing the energy bins, the modulated spectra and the LIS_isotope
        ===
        error code:
        [-1] : generic error with Modulation_Matrix_dict_Isotope
    """
    LIS_isotope_Tkin,LIS_isotope_Flux=LIS_isotope
    # ------------------------------
    # get the probability distribution function
    # ------------------------------
    try:
        # OuterEnergy_low         = data['OuterEnergy_low']
        BounduaryDistribution   = Modulation_Matrix_dict_Isotope['BounduaryDistribution']
        InputEnergy             = To_np_array([ a for a in Modulation_Matrix_dict_Isotope['InputEnergy']])
        NGeneratedPartcle       = To_np_array([ a for a in Modulation_Matrix_dict_Isotope['NGeneratedPartcle']])
        OuterEnergy             = Modulation_Matrix_dict_Isotope['OuterEnergy']
    except Exception as e: 
        print(e,file=sys.stderr)
        print(f"ERROR::Spectra() something wrong while opening the Modulation Matrix {Modulation_Matrix_dict_Isotope.keys()}", file=sys.stderr)
        return (-1.*np.ones(1),[])
    
    # ------------------------------
    # Interpolate LIS_isotope
    # ------------------------------
    # print(LIS_isotope_Tkin,LIS_isotope_Flux,InputEnergy)
    ILIS_isotope = LinLogInterpolation(LIS_isotope_Tkin,LIS_isotope_Flux,InputEnergy)
    if OuterEnergy.dtype=='float64':
        OLIS_isotope = LinLogInterpolation(LIS_isotope_Tkin,LIS_isotope_Flux,OuterEnergy)
        OENK = OuterEnergy

    #print(BounduaryDistribution)
    # ------------------------------
    # Evalaute Flux
    # ------------------------------
    UnNormFlux = np.zeros(len(InputEnergy))
    for indexTDet in range(len(InputEnergy)):
        #EnergyDetector = InputEnergy[indexTDet]
        if OuterEnergy.dtype!='float64':
            OLIS_isotope = LinLogInterpolation(LIS_isotope_Tkin,LIS_isotope_Flux,OuterEnergy[indexTDet])
            OENK = OuterEnergy[indexTDet]

        for indexTLIS_isotope in range(len(OLIS_isotope)):
            EnergyLIS_isotope = OENK[indexTLIS_isotope]
            # print(EnergyLIS_isotope)
            UnNormFlux[indexTDet]+=BounduaryDistribution[indexTDet][indexTLIS_isotope]*OLIS_isotope[indexTLIS_isotope] /beta_(EnergyLIS_isotope,T0)
    J_Mod = [ UnFlux/Npart *beta_(T,T0) for T,UnFlux,Npart in zip(InputEnergy,UnNormFlux,NGeneratedPartcle)]
    # print J_Mod
    if InputEnergy[0]>InputEnergy[-1]:
        # -- Reverse order in the LIS_isotopet
        EnergyBinning = np.array(InputEnergy[::-1])
        J_Mod         = np.array(J_Mod[::-1]) 
        LIS_isotope   = np.array(ILIS_isotope[::-1]) 
    else:
        EnergyBinning = np.array(InputEnergy[:])
        J_Mod         = np.array(J_Mod[:]) 
        LIS_isotope   = np.array(ILIS_isotope[:]) 
    return (EnergyBinning,J_Mod,LIS_isotope) 
# ------------------------------------
def Spectra_BackwardRigidity(Modulation_Matrix_dict_Isotope,LIS_isotope):
    """ Evaluate the modulated specra for a single isotope in case of SDE Monte Carlo in Rigidity
        Modulation_Matrix_dict_Isotope: dictionary, containing the matrix of the simulation for a specific isotope
        LIS_isotope: tuple, containing the energy bins and the LIS_isotope spectra for selected ion
        Returns: tuple, containing the energy bins, the modulated spectra and the LIS_isotope
        ===
        error code:
        [-1] : generic error with Modulation_Matrix_dict_Isotope
    """
    # DA SCRIVERE!
    return ([-99],[-99],[-99]) 
# ------------------------------------
def EvaluateModulation(Ion,IonLIS,Modulation_Matrix_dict,output_in_energy=True):
    """
        Evaluate the modulation of cosmic rays for a given ion species.
        Parameters:
        Ion (str): The ion species to evaluate.
        IonLIS (array-like): The local interstellar spectrum (LIS) for the ion species.
        Modulation_Matrix_dict (dict): A dictionary containing modulation matrices for different isotopes.
        output_in_energy (bool, optional): If True, the output is evaluated in kinetic energy. If False, convert to rigidity. Default is True.
        Returns:
        tuple: A tuple containing:
            - SimEnRig (array-like): The energy or rigidity binning.
            - SimFlux (array-like): The modulated flux.
            - SimLIS (array-like): The local interstellar spectrum (LIS) after modulation.
    """
    IsotopesList=findKey(Ion,Isotopes_dict)
    for iIsot  in range(len(IsotopesList)) :
        Z,A,T0,Isotope = IsotopesList[iIsot]
        print(Isotope)
        
        #print(LIS)
        lis_spectrum = GetLIS(IonLIS, Z, A)                 
        EnergyBinning,J_Mod,J_LIS=Spectra_BackwardEnergy(Modulation_Matrix_dict[Isotope],lis_spectrum,T0)
        if iIsot==0:
            SimEnRig=np.copy(EnergyBinning)
            SimFlux =np.zeros_like(EnergyBinning)
            SimLIS  =np.zeros_like(EnergyBinning)
        #print(J_LIS,J_Mod)
        if not output_in_energy:
            SimEnRig,J_Mod=Tkin2Rigi_FluxConversione(EnergyBinning,J_Mod,MassNumber=A,Z=Z)
            SimEnRig,J_LIS=Tkin2Rigi_FluxConversione(EnergyBinning,J_LIS,MassNumber=A,Z=Z)
        
        SimFlux+=J_Mod
        SimLIS+=J_LIS

    return (SimEnRig,SimFlux,SimLIS)

# ------------------------------------
if __name__ == "__main__":
    print("example of modulated spectrum for a single isotope of proton and Proton as CR-Ion")
    # Example usage of modulation funcitons
    import pickle
    from LIS import LoadLIS,GetLIS
    
    exampleLIS = LoadLIS('Librerie/esempi/GALPROP_LIS_Esempio')
    
    # load Modulation_Matrix_dict
    Monte_carlo_output_example_filename = "Librerie/esempi/Proton_RawMatrixFile.pkl"
    try:
        with open(Monte_carlo_output_example_filename, "rb") as f:
            Modulation_Matrix_dict=pickle.load(f)
    except Exception as error: 
        print(error,file=sys.stderr)
        print(f"Some error occours while opening {Monte_carlo_output_example_filename}")
        exit()

    # modulation of a single isotope
    Z = 1  # Hydrogen
    A = 1  # Atomic mass
    K = 0  # K-shell
    T0= 0.938272 # proton rest mass
    IncludeSecondaries = True
    lis_spectrum = GetLIS(exampleLIS, Z, A, K, IncludeSecondaries)
    EnergyBinning,ModulatedSpectra,LIS_isotope=Spectra_BackwardEnergy(Modulation_Matrix_dict['Proton'],lis_spectrum,T0)


    # modulation of a Ion (p+d)
    Ion="Proton"
    output_in_energy=True
    Energies,Jmod,JLIS=EvaluateModulation(Ion,exampleLIS,Modulation_Matrix_dict,output_in_energy)


    print("Energy\t Mod. spectra of p (LIS)\t-\tEnergy \t Mod. spectra of p+d (LIS)")
    for EEE,JJJ,LLL,eee,jjj,lll in zip(EnergyBinning,ModulatedSpectra,LIS_isotope,Energies,Jmod,JLIS):
        print(f"{EEE:.4f} GeV/n\t {JJJ:.3e} [m^2 s st GeV/n]^-1 (unmod: {LLL:.4e}) - {eee:.4f} GeV/n\t {jjj:.3e} [m^2 s st GeV/n]^-1 (unmod: {lll:.4e})")