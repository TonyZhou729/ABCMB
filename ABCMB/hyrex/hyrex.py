import numpy as np
import jax.numpy as jnp
from jax import jit, config, lax, grad
from jax import debug
import equinox as eqx
from diffrax import Kvaerno3

from functools import partial

from .hydrogen import hydrogen_model
from .helium import helium_model
from .array_with_padding import array_with_padding
import time
config.update("jax_enable_x64", True)

class recomb_model(eqx.Module):
    """
    Complete recombination model implementation.

    Combines helium and hydrogen recombination calculations with
    reionization modeling to compute full ionization history.

    Methods:
    --------
    get_history : Compute complete recombination and reionization history (units: dimensionless)
    """

    integration_spacing : jnp.float64
    lna_axis_full : jnp.array
    concrete_axis_size : jnp.array
    concrete_axis_size_postSahaHe : jnp.array

    twog_redshift : jnp.float64
    He4equil_redshift : jnp.float64
    idx_late : jnp.array
    idx_4He_equil : jnp.array

    def __init__(self,integration_spacing = 5.0e-4, Nsteps=800, Nsteps_postSahaHe=4000, z0=8000., z1=0.):
        """
        Initialize complete recombination model.

        Sets up time grids and parameters for helium recombination,
        hydrogen recombination, and reionization phases.

        Parameters:
        -----------
        integration_spacing : float, optional
            Step size for integration (default: 5.0e-4)
        Nsteps : int, optional
            Maximum integration steps (default: 800)
        Nsteps_postSahaHe : int, optional
            Maximum steps for post-Saha helium phase (default: 4000)
        z0 : float, optional
            Initial redshift (default: 8000.)
        z1 : float, optional
            Final redshift (default: 0.)
        """
        self.integration_spacing = integration_spacing

        # Define time axes
        self.lna_axis_full  = jnp.arange(-jnp.log(1+z0), -jnp.log(1+z1), self.integration_spacing)
        self.concrete_axis_size = jnp.zeros(Nsteps)
        self.concrete_axis_size_postSahaHe = jnp.zeros(Nsteps_postSahaHe)

        self.twog_redshift = 701.
        self.He4equil_redshift = 3601. # generous

        self.idx_4He_equil = jnp.where(self.lna_axis_full <= -jnp.log(self.He4equil_redshift))[0]
        self.idx_late  = jnp.where(self.lna_axis_full >= -jnp.log(self.twog_redshift))[0]

    def __call__(self, BG,  z_reion = 11, Delta_z_reion = 0.5, rtol=1e-6, atol=1e-9,solver=Kvaerno3(),max_steps=1024):
        """
        Compute complete recombination and reionization history.

        Parameters:
        -----------
        BG : cosmology.Background
            Background cosmology module
        z_reion : float, optional
            Reionization redshift (default: 11)
        Delta_z_reion : float, optional
            Reionization transition width (default: 0.5)
        rtol : float, optional
            Relative tolerance for ODE solver (default: 1e-6)
        atol : float, optional
            Absolute tolerance for ODE solver (default: 1e-9)
        solver : diffrax.Solver, optional
            ODE solver instance (default: Kvaerno3())
        max_steps : int, optional
            Maximum solver steps (default: 1024)

        Returns:
        --------
        tuple
            (xe_full_reion, lna_full, Tm, lna_Tm) - complete ionization history
            with reionization, log scale factor, matter temperature, and temperature grid
        """
        return self.get_history(BG, z_reion, Delta_z_reion, rtol, atol, solver, max_steps)
    
    def get_history(self, BG,  z_reion = 11, Delta_z_reion = 0.5,rtol=1e-6, atol=1e-9,solver=Kvaerno3(),max_steps=1024):
        """
        Compute complete recombination and reionization history.

        Combines helium recombination, hydrogen recombination, and
        reionization to produce complete ionization fraction evolution.

        Parameters:
        -----------
        BG : cosmology.Background
            Background cosmology module
        z_reion : float, optional
            Reionization redshift (default: 11)
        Delta_z_reion : float, optional
            Reionization transition width (default: 0.5)
        rtol : float, optional
            Relative tolerance for ODE solver (default: 1e-6)
        atol : float, optional
            Absolute tolerance for ODE solver (default: 1e-9)
        solver : diffrax.Solver, optional
            ODE solver instance (default: Kvaerno3())
        max_steps : int, optional
            Maximum solver steps (default: 1024)

        Returns:
        --------
        tuple
            (xe_full_reion, lna_full, Tm, lna_Tm) containing complete ionization
            fraction evolution with reionization, log scale factor grid,
            matter temperature, and temperature grid
        """

        lna_axis_4Heequil  = self.lna_axis_full[self.idx_4He_equil]
        lna_axis_late  = self.lna_axis_full[self.idx_late]

        xe_4He, lna_4He = helium_model(lna_axis_4Heequil)(BG)
        xe_full, lna_full, Tm, lna_Tm = hydrogen_model(xe_4He,lna_4He,lna_axis_late,lna_4He.lastval)(BG)

        ### Hydrogen Reionization ###
        # We patch a simple tanh solution to the tail of the electron fraction result.
        fHe = BG.params['YHe'] / 4 / (1-BG.params['YHe'])
        z = 1/jnp.exp(lna_full.arr) - 1
        y = (1+z)**(3./2)

        y_reion = (1+z_reion)**(3./2)
        Delta_y_reion = 3./2 * jnp.sqrt(1+z_reion) * Delta_z_reion
        tanh_arg = (y_reion - y) / Delta_y_reion

        xe_reion_correction = (1+fHe)/2 * (1 + jnp.tanh(tanh_arg))
        xe_full_arr = xe_reion_correction + xe_full.arr 
        xe_full_reion = array_with_padding(xe_full_arr)
        ### End of Hydrogen Reionization ###

        # best return the whole array-with-padding object 
        # so we can interpolate over the padding
        return (xe_full_reion, lna_full, Tm, lna_Tm)  