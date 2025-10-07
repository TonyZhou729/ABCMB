from jax import jit, config
import jax.numpy as jnp
import equinox as eqx
import diffrax

from .hyrex import hyrex
from . import cosmology, perturbations, spectrum
from . import constants as cnst
from . import AbstractSpecies as AS

config.update("jax_enable_x64", True)

class Precision(eqx.Module):
    """
    Computational precision parameters.

    Contains constants for transfer function calculations and
    line-of-sight integrals.

    Attributes:
    -----------
    T0_largek_cut : float
        T0 term k-space cutoff (units: Mpc^{-1})
    T1_largek_cut : float
        T1 term k-space cutoff (units: Mpc^{-1})
    T2_largek_cut : float
        T2 term k-space cutoff (units: Mpc^{-1})
    E0_largek_cut : float
        E-mode polarization k-space cutoff (units: Mpc^{-1})
    tau_c_over_tau_h_largex_cut : float
        Thomson scattering time threshold
    jl_smallx_cut : float
        Spherical Bessel function cutoff threshold
    """

    ### TRANSFER FUNCTION RELATED ###

    # MULTIPOLE CUT APPROXIMATION #
    # These are the CLASS defaults, all for SCALAR MODES.
    # Cuts the line of sight integral over k, for a given ell mode, equal to kmax = l/rA_rec + X_largek_cut,
    # where rA_rec is the comoving sound horizon at recombination
    T0_largek_cut : float = 0.15 # T0 term of the temperature source function, contains SW and a part of the ISW effects.
    T1_largek_cut : float = 0.04 # T1 term of the temperature source function, contains the remaining ISW effect.
    T2_largek_cut : float = 0.15 # T2 term of the temperature source function
    E0_largek_cut : float = 0.11 # E-mode polarization term.

    # TIME CUT APPROXIMATION #
    tau_c_over_tau_h_largex_cut : float = 0.008 # Start the lna integration at a time when aH x tau_c = tau_c_over_tau_h_largex_cut.
    jl_smallx_cut               : float = 1.e-5 # Stop the upperbound of the lna integration at a time when jl(x) < jl_smallx_cut, where x=k(tau0-tau).

class Model(eqx.Module):
    """
    Model configuration and computation manager.

    Creates instances of fluid species based on user input and organizes
    them for computation. Manages the full pipeline from background
    evolution through CMB power spectrum computation.

    Methods:
    --------
    run_cosmology : Compute CMB angular power spectra
    get_PTBG : Get perturbation table and background cosmology
    get_BG : Get background cosmology
    add_derived_parameters : Compute derived parameters
    """

    RM : hyrex.recomb_model
    #PE : perturbations.PerturbationEvolver
    SS : spectrum.SpectrumSolver

    species_list       : tuple = ()
    perturbations_list : tuple = ()

    ### ADDING SPECIES: add has_ parameter and add condition to append to tuple.
    # In the init, all species that are present within the model should be set to True.
    # All couplings present between species should be set to true. 
    def __init__(self,
                 ellmin = 2,
                 ellmax = 2500,
                 lensing = False,
                 has_MasslessNeutrinos=False, 
                 has_MassiveNeutrinos=False): 

        self.SS = spectrum.SpectrumSolver(ellmin, ellmax, lensing, switch_sw=1., switch_isw=1., switch_dop=1., switch_pol=1.)

        diffrax_vector_idx = 2 # The first two indices (0 and 1) are always reserved for the metric perturbations.
        perturbations_list = ()

        cold_dark_matter = AS.ColdDarkMatter(diffrax_vector_idx)
        self.species_list = self.species_list + (cold_dark_matter,)
        diffrax_vector_idx += cold_dark_matter.num_ell_modes # Add to total length of Diffrax vector

        dark_energy = AS.DarkEnergy()
        self.species_list = self.species_list + (dark_energy,)

        baryon = AS.Baryon(diffrax_vector_idx, dark_energy)
        diffrax_vector_idx += baryon.num_ell_modes # Add to total length of Diffrax vector

        photon = AS.Photon(diffrax_vector_idx, baryon)
        diffrax_vector_idx += photon.num_ell_modes # Add to total length of Diffrax vector

        baryon = eqx.tree_at(lambda b : b.photon, baryon, photon)
        self.species_list = self.species_list + (baryon, photon,)

        if has_MasslessNeutrinos:
            massless_neutrinos = AS.MasslessNeutrinos(diffrax_vector_idx)
            self.species_list   = self.species_list + (massless_neutrinos,)
            diffrax_vector_idx += massless_neutrinos.num_ell_modes # Add to total length of Diffrax vector

        if has_MassiveNeutrinos:
            massive_neutrinos = AS.MassiveNeutrinos(diffrax_vector_idx)
            self.species_list   = self.species_list + (massive_neutrinos,)
            diffrax_vector_idx += massive_neutrinos.num_ell_modes # Add to total length of Diffrax vector

        for species in self.species_list:
            if isinstance(species, AS.AbstractPerturbedFluid):
                self.perturbations_list = self.perturbations_list + (species, )

        self.RM = hyrex.recomb_model()
        #self.PE = perturbations.PerturbationEvolver(perturbations_list)
    
    @jit
    def run_cosmology(self, params : dict):
        """
        Compute CMB angular power spectra for given parameters.

        Runs the full pipeline from background evolution through
        perturbation integration to CMB power spectrum computation.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        tuple
            (ℓ values, (C_ℓ^TT, C_ℓ^TE, C_ℓ^EE)) for computed multipoles
        """
        # Set up the parameter handler object for the current run given the set
        # of parameters. This is to be passed to individual species instances to
        # calculate relevant quantities such as the energy density.
        #PT, BG = self.get_PTBG(params)
        ### COMPUTING POWER SPECTRA ###
        #idxs = jnp.arange(18, 30) # Only compute at tabulated l positions. Future: Adjust l_max # CG: 80 for l = 2000
        #return spectrum.bessel_l_tab[idxs], self.SS.get_Cl(idxs, PT, BG)
        PT, BG = self.get_PTBG(params)
        #return self.SS.Pk_lin(PT.k, 0., PT, BG)
        #return PT.delta_b
        Cls = self.SS.get_Cl(PT, BG)
        return Cls

    # @jit
    def get_PTBG(self, params : dict):
        """
        Get perturbation table and background.

        Computes background and evolves perturbations for the given parameters.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        tuple
            (PerturbationTable, Background) objects
        """
        BG = self.get_BG(params)
        PE = perturbations.PerturbationEvolver(self.perturbations_list, BG)
        
        # Specify whether to use full_evolution() or full_evolution_scan()
        #PT = PE.full_evolution()
        PT = PE.full_evolution_scan()
        return PT, BG

    # @jit
    def get_BG(self, params : dict):
        """
        Get background for given parameters.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        cosmology.Background
            Background object
        """
        params = self.add_derived_parameters(params)
        BG = cosmology.Background(params, self.species_list, self.RM)
        return BG

    def add_derived_parameters(self, params : dict) -> dict:
        """
        Compute derived parameters.

        Calculates derived parameters from the fundamental parameters,
        including density parameters and ratios.

        Parameters:
        -----------
        params : dict
            Input parameters

        Returns:
        --------
        dict
            Extended parameter dictionary with derived quantities
        """
        params['omega_m']      = params['omega_cdm'] + params['omega_b']
        params['R_b']          = params['omega_b'] / params['omega_m']
        params['omega_g']      = 8. * jnp.pi**3 * cnst.G / 45. / cnst.H0_over_h**2 / cnst.hbar**3 / cnst.c**3 * params['TCMB0']**4
        params['H0']           = params['h'] * cnst.H0_over_h
        params['N_ur']         = params['Neff'] - (params['T_ncdm'] / params['TCMB0'])**4 / (4. / 11.)**(4. / 3.) * params['N_ncdm']
        params['omega_nu']     = 7. / 8. * params['N_ur'] * (4. / 11.)**(4. / 3.) * params['omega_g']
        params['omega_r']      = params['omega_g'] + params['omega_nu']
        params['R_nu']         = jnp.where(params['omega_r'] > 0.0, params['omega_nu'] / params['omega_r'], 0.0)
        params['omega_Lambda'] = params['h']**2 - params['omega_r'] - params['omega_m']
        return params