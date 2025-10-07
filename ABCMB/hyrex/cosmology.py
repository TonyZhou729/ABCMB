from jax import config
import jax.numpy as jnp
config.update("jax_enable_x64", True)

c           = jnp.float64(29979245800.)           # Speed of light in cm s^{-1}
hbar        = jnp.float64(6.582119569509e-16)     # hbar in eV s
G           = jnp.float64(1.18980205e-40)         # Newton's gravitational constant (G/c^2), in cm^3 eV^{-1} s^{-2}
kB          = jnp.float64(8.617343e-5)            # Boltzmann constant in eV / K
mp          = jnp.float64(0.938271999e+09)        # Proton rest mass, in eV
me          = jnp.float64(510998.9461)            # Electron rest mass, in eV
TCMB_today  = jnp.float64(2.34865418e-4)          # CMB temperature today in eV.
conv_factor = jnp.float64(3.2407792894443648e-18) # (100 km/s/Mpc) in units of s^{-1}
mH          = mp+me+13.598286071938324            # Neutral hydrogen atom rest mass, in eV
mu_e        = mp*me/mH                            # Reduced mass of proton-electron system, in eV

def a(z):
    """
    Converts redshift to scale factor.

    Dimensions: None

    Parameters
    ----------
    z : float/jnp.array
        Requested redshift(s)

    Returns
    -------
    a : float/jnp.array
        Scale factor
    """
    return 1. / (1.+z)

def omega_rad0(Neff=3.044):
    """
    Calculates radiation density today.

    Dimensions: None

    Parameters
    ----------
    Neff : float
        Cosmological parameter, effective number of neutrino species

    Returns
    -------
    omega_rad0 : float
        The radiation density today.
    """
    # CMB photon energy density, in eV cm^{-3}
    rho_g0 = jnp.pi**2/15./hbar**3/c**3 * TCMB_today**4
    
    # Neutrino energy density today
    rho_nu0 = Neff*(7./8. * (4./11.)**(4./3.)) * rho_g0

    # Total radiation density
    rho_r0 = rho_g0 + rho_nu0

    # Divide by rho critical today to yield the dimensionless density param
    omega_rad0 = 8.*jnp.pi*G/3./conv_factor**2 * rho_r0

    return omega_rad0

def Hubble(z, h, omega_b, omega_cdm, omega_rad):
    """
    Computes the Hubble constant at given redshift and values of relevant cosmology parameters.

    Dimensions: s^{-1}

    WARNING: Assumes flatness (i.e. Om_m + Om_r + Om_Lambda = 1)

    Parameters
    ----------
    z : float/jnp.array
        Requested redshift(s)
    h : float
        Cosmological parameter, reduced Hubble parameter today defined as H0 / (100 km/s/Mpc)
    omega_b : float
        Cosmological parameter, baryon density fraction today.
    omega_cdm : float
        Cosmological parameter, cold dark matter density fraction today.
    omega_rad : float
        Cosmological parameter, radiation density fraction today.
    
    Returns
    -------
    H : float/jnp.array
        Hubble parameter at requested redshifts, in s^{-1}.
    """
    
    # Total matter density (same redshift for both)
    omega_m = omega_b + omega_cdm

    # Dark energy density
    omega_Lambda = h**2 - omega_m - omega_rad
    
    # Friedmann's equation
    H = conv_factor*jnp.sqrt(omega_rad/a(z)**4 + omega_m/a(z)**3 + omega_Lambda)

    return H

def nH(z, omega_b, YHe):
    """
    Computes the total hydrogen number density at redshift z.
    
    Dimensions: cm^{-3}

    Parameters
    ----------
    z : float/jnp.array
        Requested redshift(s).
    omega_b : float
        Cosmological parameter, baryon density fraction today.
    YHe : float
        Helium mass fraction.

    Returns
    -------
    nH : float
        Hydrogen number density at redshift z.
    """
    return (1-YHe) * 3 * omega_b * conv_factor**2 / 8 / jnp.pi / G / mH / a(z)**3

def TCMB(z):
    """
    Computes the CMB temperature at redshift z.

    Dimensions: eV

    Parameters
    ----------
    z : float/jnp.array
        Requested redshift(s).

    Returns
    -------
    TCMB : float/jnp.array
        CMB temperature.
    """
    return TCMB_today / a(z)


