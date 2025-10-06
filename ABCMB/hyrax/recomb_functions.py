import numpy as np
import jax.numpy as jnp
from jax import config, pure_callback, devices, device_put
from jax.scipy.ndimage import map_coordinates
from ABCMB import constants as cnst
import jax.experimental.host_callback as hcb
import os
file_dir = os.path.dirname(__file__)
config.update("jax_enable_x64", True)

#Tabulated values of effective recombination coefficients to interpolate.
R_tab     = jnp.array(np.loadtxt(file_dir+"/tabs/R_inf.dat"))

#Tabulated values of 2s-2p transition rates to interpolate.
alpha_tab = jnp.array(np.loadtxt(file_dir+"/tabs/Alpha_inf.dat"))

try:
    gpus = devices('gpu')
    R_tab = device_put(
        R_tab, device=gpus[0])
    alpha_tab = device_put(
        alpha_tab, device=gpus[0])
except: 
    pass

# File handling and interpolating related constants.
# Do not change these unless something about the tabulated files have changed.
TR_MIN      = 0.004 # Lower bound of radiation temperature axis, in eV.
TR_MAX      = 0.4   # Upper bound of radiation temperature axis, in eV.
NTR         = 100   # Number of points in radiation temperature axis.

T_RATIO_MIN = 0.1   # Lower bound of temperature ratio axis, defined as min(Tm/Tr, Tr/Tm).
T_RATIO_MAX = 1.0   # Upper bound of temperature ratio axis, defined as min(Tm/Tr, Tr/Tm).
NTM         = 40    # Number of points in temperature ratio axis

TR_axis = jnp.linspace(jnp.log(TR_MIN), jnp.log(TR_MAX), NTR)
T_RATIO_axis = jnp.linspace(T_RATIO_MIN, T_RATIO_MAX, NTM)
A2s_table = alpha_tab.reshape((NTR, NTM, 4))
A2p_table = alpha_tab.reshape((NTR, NTM, 4))

def Gamma_compton(xe, TCMB, YHe):
    """
    Computes the Compton scattering rate. See Eq.(2) of 1904.09296

    Dimensions: s^{-1}

    Parameters
    ----------
    xe : float
        Electron fraction.
    TCMB : float
        CMB temperature, in eV.
    YHe : float
        Helium mass fraction. 
    
    Returns
    -------
    GammaC : float
        The Compton scattering rate
    """
    # Helium to Hydrogen number density ratio. Here we assumed that all Helium are Helium-4.
    # 1) n_H = (1-YHe)*n_b
    # 2) n_He = (YHe/4)*n_b
    FHe = YHe/4/(1-YHe)
   
    GammaC = xe / (1.+FHe+xe) * 8.*cnst.thomson_xsec*(4.*cnst.stef_bolt)*TCMB**4 / (3.*cnst.me)
    return GammaC

def xe_Saha(TCMB, nH):
    """
    Computes the free electron fraction in the Saha equilibrium approximation, valid
    at early times when matter and photons are in Chemical Equilibrium.

    Parameters
    ----------
    TCMB : float
        CMB temperature, in eV
    nH : float
        Current Hydrogen number density, in cm^{-3}

    Returns
    -------
    xe_saha : float
        Saha equilibrium prediction of free electron fraction.

    s : float
        xe_saha^2/(1-xe_saha)
    """
    ge = (2.*jnp.pi*cnst.mu_e*TCMB)**(3./2.) / (2.*jnp.pi*cnst.hbar*cnst.c)**3. / nH
    s = ge * jnp.exp(-cnst.rydberg/TCMB)
    xe_saha = (jnp.sqrt(s**2.+4.*s) - s) / 2.
    return xe_saha, s

def effective_coefficients(TCMB, Tm, H, nH, x1s):
    """
    Computes the effective coefficients for
        Case-B recombination (A2s, A2p)
        Case-B photoionization (B2s, B2p)
        Generalized Peebles factors (C2s, C2p)
    in the effetive four-level-atom strategy outlined by arXiv:1011.3758.

    Dimensions:
        A2s, A2p : cm^3 s^{-1}
        B2s, B2p : s^{-1}
        C2s, C2p : None

    Parameters
    ----------
    TCMB : float
        Current CMB temperature, in eV.
    Tm : float
        Current matter temperature, in eV.
    H : float
        Current Hubble parameter, in s^{-1}.
    nH : float
        Current proton (free+bound) number density, in cm^{-3}.
    x1s : float
        Fraction of proton in the 1s bound state.    

    Returns
    -------
    coefficients : tuple (float, float, float, float, float, float, float, float)
        A tuple of the six relevant effective coefficients for the four-level-atom. 
    """
    """ Step 0: Set up interpolation environment. """

    # Determine which columns of alpha_tab to use based on TCMB, Tm
    # Always take the smaller of the two in numerator, such that ratio is < 1.
    
    # The if statements are only necessary if Tm > TCMB ever, but that seems not realistic?
    #T_RATIO_now = jnp.where(Tm < TCMB, Tm/TCMB, TCMB/Tm) # The input point to interpolate at.
    #A2s_column = jnp.where(Tm < TCMB, 0, 2)
    #A2p_column = A2s_column + 1
    T_RATIO_now = Tm/TCMB
    A2s_column = 0
    A2p_column = 1

    # Determine the array position of the requested temperatures relative to the tabulated axis.
    # For instance, if axis is [0, 2, 4, 6, 8], requesting 3 should be position 1.5
    # This is needed to use ndimage.map_coordinates
    TCMB_index = jnp.interp(jnp.log(TCMB), TR_axis, jnp.arange(TR_axis.size))
    T_RATIO_index = jnp.interp(T_RATIO_now, T_RATIO_axis, jnp.arange(T_RATIO_axis.size))
    
    """ Step 1: Obtain recombination coefficients A2s, A2p by interpolating tabulated alpha. """    
    A2s = jnp.exp(map_coordinates(jnp.log(A2s_table[:, :, A2s_column]), [TCMB_index, T_RATIO_index], order=1))
    A2p = jnp.exp(map_coordinates(jnp.log(A2p_table[:, :, A2p_column]), [TCMB_index, T_RATIO_index], order=1))

    """ Step 2: Obtain photoionization coefficients B2s, B2p from A2s, A2p via detailed balance. """
    # See equation (22) of arXiv:1006.1355 for detail. 
    # nl state energy -EI/n^2, where EI is the hydrogen ionization energy, for n=2.
    E2 = -cnst.rydberg / 4.
    # equation (C7) of arXiv:1006.1355, times c^3, dimensions of cm^3
    q = (2*jnp.pi*cnst.mu_e*TCMB / (2*jnp.pi*cnst.hbar)**2)**(3/2) / cnst.c**3
    
    B2s = jnp.exp(E2/TCMB) * q * jnp.exp(map_coordinates(jnp.log(A2s_table[:, :, A2s_column]), [TCMB_index, NTM-1], order=1))
    B2p = (1./3.)*jnp.exp(E2/TCMB) * q * jnp.exp(map_coordinates(jnp.log(A2p_table[:, :, A2p_column]), [TCMB_index, NTM-1], order=1))

    """ Step 3: Obtain 2p->2s transition rate by interpolating tabulated R, and 2s->2p rate via detailed balance. """ 
    R2p2s = jnp.exp(jnp.interp(jnp.log(TCMB), TR_axis, jnp.log(R_tab)))
    R2s2p = 3.*R2p2s # See equation (21) of arXiv:1006.1355
    
    """ Step 4: Obtain generalized Peebles C factors for 2s, 2p states """    
    # Rate of escape of lyman-alpha photons, multiplied by the bound proton fraction to regulate possible divergence.
    R_Lya_times_x1s = 8.*jnp.pi*H / (3.*nH*(cnst.c/cnst.lya_freq)**3) 
    Gamma2s = B2s + R2s2p + cnst.R2s1s # Inverse lifetime of 2s state
    Gamma2p_times_x1s = (B2p + R2p2s)*x1s + R_Lya_times_x1s # Inverse lifetime of 2p state, with x1s factor.
    C2s = (cnst.R2s1s + R2s2p*R_Lya_times_x1s/Gamma2p_times_x1s) \
        / (Gamma2s - x1s*R2s2p*R2p2s/Gamma2p_times_x1s)
    C2p = (R_Lya_times_x1s + x1s*R2p2s*cnst.R2s1s/Gamma2s) \
        / (Gamma2p_times_x1s - x1s*R2p2s*R2s2p/Gamma2s)
    
    return jnp.array([A2s, A2p, B2s, B2p, C2s, C2p, R2p2s, R2s2p])

