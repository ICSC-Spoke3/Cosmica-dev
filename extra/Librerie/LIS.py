import astropy.io.fits as pyfits # per elaborare i fits file
import numpy as np
import sys
"""LIS.py - Library for managing Local Interstellar Spectrum (LIS)
Author: Stefano Della Torre
Date: 2024/12/24
Version: 1.0
This library provides functions to load and retrieve Local Interstellar Spectrum (LIS) data from FITS files.
Functions:
----------
1. LoadLIS(InputLISFile='../LIS/ProtonLIS_ApJ28nuclei.gz'):
    Load LIS from a FITS file.
    Parameters:
        InputLISFile (str): Path to the FITS file.
    Returns:
        tuple: A tuple containing:
            - LIS_Tkin (numpy.ndarray): Energy bins in GeV/n.
            - LIS_Flux (dict): Dictionary containing the LIS for all species.
                ParticleFlux[Z][A][K] contains the particle flux for all considered species.
                The galprop convention is used where for the same combination of Z, A, K,
                the first entries are secondaries, and the latter are primaries.
2. GetLIS(LIS, Z, A, K=0, IncludeSecondaries=True):
    Return the LIS for the selected isotope.
    Parameters:
        LIS (tuple): A tuple containing the energy bins and the LIS dictionary.
        Z (int): Atomic number.
        A (int): Atomic mass.
        K (int): K-shell.
        IncludeSecondaries (bool): If True, include secondary spectra in the output.
    Returns:
        tuple: A tuple containing:
            - TK_bin (numpy.ndarray): Energy bins.
            - TK_LISSpectra (numpy.ndarray): LIS for the selected isotope.
        Error return status:
            - [-1]: Z does not exist in LIS dictionary.
            - [-2]: A does not exist in LIS dictionary for the selected Z.
Example Usage:
--------------
1. Load LIS data from a FITS file:
2. Print the range of energies and the number of bins:
3. Print the keys of LIS_Flux in a table format:
4. Retrieve and print the LIS for a specific isotope:
"""

# LIS.py - libreria per la gestione dei LIS (local interstellar spectrum)
# 2024/12/24 - Stefano Della Torre - Versione 1.0

#-------------------------------------------------------------------
def LoadLIS(InputLISFile='../LIS/ProtonLIS_ApJ28nuclei.gz'):
    """Load LIS from a fits file
    InputLISFile: string, path to the fits file
    Returns: dictionary, containing the LIS for all species 
        ParticleFlux[Z][A][K] contains the particle flux for all considered species 
        galprop convention wants that for same combination of Z,A,K 
        firsts are secondaries, latter Primary
    """
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
    energy = energy/1e3 # convert energy scale from MeV/n to GeV/n
    #if A>1:
    #  energy = energy/float(A)
    hdulist.close()
    #LISSpectra = [ 0 for T in energy]
    LIS_Tkin=energy
    LIS_Flux=ParticleFlux  
    return (LIS_Tkin,LIS_Flux)

# -------------------------------------------------------------------
def GetLIS(LIS,Z, A, K=0, IncludeSecondaries=True):
    """Return the LIS for the selected Isotope,
        LIS: dictionary, containing the LIS for all isotopes
        Z: int, atomic number
        A: int, atomic mass
        K: int, K-shell
        IncludeSecondaries: bool, if True, include secondary spectra in output
        Returns: tuple, containing the energy bins and the LIS
        
        ---------- error return status ---------
            [-1]: Z does not exist in LIS dictionary
            [-2]: A does not exist in LIS dictionary for selected Z
    """
    # -------- init variables
    TK_bin,ParticleFlux=LIS
    # == check of Z is available
    if (Z not in ParticleFlux):     # Return -1 if Z does not exist
        print(f"Error: Z={Z} does not exist in LIS dictionary", file=sys.stderr)
        return (-1.*np.ones(1),[])
    # == check of A is available
    if (A not in ParticleFlux[Z]):     # Return -2 if A does not exist for selected Z
        print(f"Error: A={A} does not exist in LIS dictionary for Z={Z}", file=sys.stderr)
        return (-2.*np.ones(1),[])
    # ==
    TK_LISSpectra= (ParticleFlux[Z][A][K])[-1]   # the primary spectrum is always the last one (if exist)
    if IncludeSecondaries:                       # include secondary spectra
        for SecInd in range(len(ParticleFlux[Z][A][K])-1):
            TK_LISSpectra= TK_LISSpectra+(ParticleFlux[Z][A][K])[SecInd] # Sum All secondaries
    return (TK_bin, TK_LISSpectra) 

# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage of LoadLIS
    LIS_Tkin, LIS_Flux = LoadLIS('Librerie/esempi/GALPROP_LIS_Esempio')
    
    # a) Print the range of energies in LIS_Tkin and the number of bins
    print(f"Energy range: {LIS_Tkin[0]} - {LIS_Tkin[-1]} GeV/n")
    print(f"Number of bins: {len(LIS_Tkin)}")
    
    # b) Print the keys of LIS_Flux in a table format
    print("Z\tA\tK")
    for Z in LIS_Flux:
        for A in LIS_Flux[Z]:
            for K in LIS_Flux[Z][A]:
                print(f"{Z}\t{A}\t{K}")

    # Example usage of GetLIS
    Z = 1  # Hydrogen
    A = 1  # Atomic mass
    K = 0  # K-shell
    IncludeSecondaries = True

    energy_bins, lis_spectrum = GetLIS((LIS_Tkin, LIS_Flux), Z, A, K, IncludeSecondaries)
    
    if len(energy_bins) > 1:
        print(f"LIS for Z={Z}, A={A}, K={K}:")
        for e, flux in zip(energy_bins, lis_spectrum):
            print(f"Energy: {e:12.5f} GeV/n, Flux: {flux:.3e} (s sr m^2 GeV)^-1")
    else:
        print("Error retrieving LIS")