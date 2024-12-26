import numpy as np
from scipy.interpolate import interp1d 
"""
Utility functions for various scientific computations and file operations.

Functions:
    To_np_array(v):
        Converts the input to a numpy array if it is not already one.

    mkdir_p(path):
        Creates a directory at the specified path, similar to the 'mkdir -p' command.

    beta_(T, T0):
        Evaluates the beta factor from kinetic energy T and rest mass energy T0.

    Rigidity(T, MassNumber=1.0, Z=1.0):
        Evaluates the rigidity from kinetic energy T, mass number, and charge number.

    Energy(R, MassNumber=1.0, Z=1.0):
        Evaluates the kinetic energy from rigidity, mass number, and charge number.

    dT_dR(T=1, R=1, MassNumber=1.0, Z=1.0):
        Computes the flux conversion factor from rigidity to kinetic energy.

    dR_dT(T=1, R=1, MassNumber=1.0, Z=1.0):
        Computes the flux conversion factor from kinetic energy to rigidity.

    Tkin2Rigi_FluxConversione(Xval, Spectra, MassNumber=1.0, Z=1.0):
        Converts flux from kinetic energy to rigidity.

    LinLogInterpolation(vx, vy, Newvx):
        Performs linear interpolation in log-log scale for given x and y arrays using new x bins.

    plot_ascii(x, y):
        Plots an XY graph in ASCII art.
"""
# general purpose, useful functions

# ========= convert to np.array =============
def To_np_array(v):
    if not isinstance(v,(np.ndarray,)):
        v = np.asarray(v)
    return v

# ========= create a folder =============
def mkdir_p(path):
    import os       # OS directory manager
    import errno    # error names
    # --- this function create a folder and check if this already exist ( like mkdir -p comand)
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass

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

# ======= plot XY graph in ascii =========
def plot_ascii(x, y):
    max_y = max(y)
    #min_y = min(y)
    maxhigh = 10  # altezza massima del grafico
    print(f"^{max_y}")
    for yi in reversed(range(0, maxhigh+1)):
        if yi !=0 :
            line = '|'
            for xi in range(len(x)):
                if yi <= y[xi]/max_y*maxhigh:
                    line += '*'
                else:
                    line += ' '
        else:
            line = '+'
            for xi in range(len(x)):
                line += '-'
        print(line)
    print(f"0")