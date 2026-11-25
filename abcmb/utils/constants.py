import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

"""
Fundamental constants
"""
c            = jnp.float64(29979245800.)           # Speed of light in cm s^{-1}
c_Mpc_over_s = jnp.float64(9.71561e-15)            # Speed of light in Mpc s^{-1}
H0_over_h    = jnp.float64(3.24078e-18)            # 100 km/s/Mpc in 1/s
hbar         = jnp.float64(6.582119569509e-16)     # hbar in eV s
G            = jnp.float64(1.18980205e-40)         # Newton's gravitational constant (G/c^2), in cm^3 eV^{-1} s^{-2}
kB           = jnp.float64(8.617343e-5)            # Boltzmann constant in eV / K

"""
Cosmology / particle physics related
"""
mp           = jnp.float64(938271999.)             # Proton rest mass, in eV
mn           = jnp.float64(939565413.)             # Neutron rest mass, in eV
me           = jnp.float64(510998.9461)            # Electron rest mass, in eV
TCMB_today   = jnp.float64(2.34865418e-4)          # CMB temperature today in eV.
conv_factor  = jnp.float64(3.2407792894443648e-18) # (100 km/s/Mpc) in units of s^{-1}
mH           = mp+me+jnp.float64(13.5982860719383) # Neutral hydrogen atom rest mass, in eV
mHe          = jnp.float64(3.72839e9)              # Helium-4 rest mass, in eV
mu_e         = mp*me/mH                            # Reduced mass of proton-electron system, in eV

"""
Recombination related
"""
E21          = jnp.float64(10.198714553953742)     # Energy difference in n=1, 2 for hydrogen, in eV.
E31          = jnp.float64(12.087365397278509)     # Energy difference in n=1, 3 for hydrogen, in eV.
E41          = jnp.float64(12.748393192442178)     # Energy difference in n=1, 4 for hydrogen, in eV.
E32          = jnp.float64(1.8886508433247664)     # Energy difference in n=2, 3 for hydrogen, in eV.
E42          = jnp.float64(2.5496786384884356)     # Energy difference in n=2, 4 for hydrogen, in eV.
rydberg      = jnp.float64(13.598286071938324)     # Ionization energy of hydrogen, in eV
lya_eng      = rydberg*3./4.                       # Lyman-alpha transition energy, in eV
lya_freq     = lya_eng / (2.*jnp.pi*hbar)          # Lyman-alpha transition frequency, in s^{-1}
thomson_xsec = jnp.float64(6.652458734e-25)        # Thomson cross section, in cm^2
stef_bolt    = jnp.pi**2 / (60.*hbar**3*c**2)      # Stefan-Boltzmann constant, eV^{-3} cm^{-2} s^{-1}
R2s1s        = jnp.float64(8.2206)                 # 2s to 1s transition rate, in s^{-1}