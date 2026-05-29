import numpy as np
import jax.numpy as jnp
from jax import config
import equinox as eqx
from diffrax import Kvaerno3, ForwardMode

from functools import partial

from .hydrogen import hydrogen_model
from .helium import helium_model
from .array_with_padding import array_with_padding
from ..ABCMBTools import fast_interp
config.update("jax_enable_x64", True)


class RecombInputs(eqx.Module):
    """
    Bundle of pre-recombination background quantities sampled on a 
    fixed lna grid for computing recombination.

    Attributes
    ----------
    lna_grid : jnp.array
        Uniform log scale-factor sampling axis. (units: dimensionless)
    TCMB_arr : jnp.array
        Photon-bath temperature TCMB(lna) (units: eV)
    nH_arr : jnp.array
        Hydrogen number density nH(lna) (units: cm^-3)
    H_arr : jnp.array
        Hubble parameter H(lna) (units: s^-1)

    Methods
    -------
    TCMB : Linear interpolation of CMB temperature over lna (units: eV)
    nH : Linear interpolation of hydrogen number density over lna (units: cm^-3)
    H : Linear interpolation of Hubble over lna (units: s^-1)
    """

    lna_grid : jnp.array
    TCMB_arr : jnp.array
    nH_arr   : jnp.array
    H_arr    : jnp.array

    def TCMB(self, lna):
        """
        Linearly interpolate CMB temperature at lna.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor.

        Returns:
        --------
        float
            CMB temperature TCMB(lna) (units: eV).
        """
        return fast_interp(lna, self.lna_grid[0], self.lna_grid[-1], self.TCMB_arr)

    def nH(self, lna):
        """
        Linearly interpolate hydrogen number density at lna.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor.

        Returns:
        --------
        float
            Hydrogen number density nH(lna) (units: cm^-3).
        """
        return fast_interp(lna, self.lna_grid[0], self.lna_grid[-1], self.nH_arr)

    def H(self, lna):
        """
        Linearly interpolate Hubble parameter at lna.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor.

        Returns:
        --------
        float
            Hubble parameter H(lna) (units: s^-1).
        """
        return fast_interp(lna, self.lna_grid[0], self.lna_grid[-1], self.H_arr)

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

    z1 : jnp.float64

    twog_redshift : jnp.float64
    He4equil_redshift : jnp.float64
    idx_4He_equil : jnp.array

    adjoint : "diffrax.adjoint" = eqx.field(static=True)

    def __init__(self, integration_spacing = 5.0e-4, z0=8000., z1=0., adjoint = ForwardMode):
        """
        Initialize complete recombination model.

        Sets up time grids and parameters for helium recombination,
        hydrogen recombination, and reionization phases.

        Parameters:
        -----------
        integration_spacing : float, optional
            Step size for integration (default: 5.0e-4)
        z0 : float, optional
            Initial redshift (default: 8000.)
        z1 : float, optional
            Final redshift (default: 0.)
        adjoint : diffrax.adjoint
            Adjoint mode for diffrax solves (static field).  Defaults
            to ForwardMode.
        """
        self.integration_spacing = integration_spacing
        self.adjoint = adjoint
        self.z1 = z1

        # Define time axes
        self.lna_axis_full  = jnp.arange(-jnp.log(1+z0), -jnp.log(1+z1), self.integration_spacing)

        self.twog_redshift = 701.
        self.He4equil_redshift = 3601. # generous

        self.idx_4He_equil = jnp.where(self.lna_axis_full <= -jnp.log(self.He4equil_redshift))[0]

    def __call__(self, args, rtol=1e-6, atol=1e-9,solver=Kvaerno3(),max_steps=1024):
        """
        Compute complete recombination and reionization history.

        Parameters:
        -----------
        args : tuple
            Recombination input arrays and cosmological parameters
            (recomb_inputs, params).
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
        return self.get_history(args, rtol, atol, solver, max_steps)

    def get_history(self, args, rtol=1e-6, atol=1e-9,solver=Kvaerno3(),max_steps=1024):
        """
        Compute complete recombination and reionization history.

        Combines helium recombination, hydrogen recombination, and
        reionization to produce complete ionization fraction evolution.

        Parameters:
        -----------
        args : tuple
            Recombination input arrays and cosmological parameters
            (recomb_inputs, params).
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

        recomb_inputs, params, species_list = args
        lna_axis_4Heequil  = self.lna_axis_full[self.idx_4He_equil]

        xe_4He, lna_4He = helium_model(lna_axis_4Heequil, adjoint=self.adjoint)(args)
        xe_full, lna_full, Tm, lna_Tm = hydrogen_model(xe_4He,lna_4He,-jnp.log(1+self.z1),lna_4He.lastval,self.twog_redshift, adjoint=self.adjoint)(args)

        return (xe_full, lna_full, Tm, lna_Tm)