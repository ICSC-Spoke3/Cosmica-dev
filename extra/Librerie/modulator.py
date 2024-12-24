
### NOT FINISHED YET ###


def Spectra(RawMatrixFile,LIS,T0,Z,A):
    """ Evaluate the modulated specra for a single isotope
        RawMatrixFile: dictionary, containing the matrix of the simulation
        LIS: tuple, containing the energy bins and the LIS
        T0: float, kinetic energy of the particle at rest
        Z: int, atomic number
        A: int, atomic mass

        Returns: tuple, containing the energy bins, the modulated spectra and the LIS
    """


#     """Return the modulated spectra according to simualtion in HelModSimPath given the LIS in (LIS_Tkin,LIS_Flux)
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