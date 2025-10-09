from jax import jit, config
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np
import equinox as eqx
import diffrax

import sys
import os
file_dir = os.path.dirname(__file__)

from .hyrex import hyrex
from . import cosmology, perturbations, spectrum
from . import constants as cnst
from . import AbstractSpecies as AS
from .ABCMBTools import bilinear_interp

from .linx.background import BackgroundModel
from .linx.abundances import AbundanceModel
from .linx.nuclear import NuclearRates

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

    bbn_type                : str = ""
    linx_reaction_net       : str = ""
    
    PArthENoPE_CLASS_table  : Array #= eqx.field(converter=jnp.asarray)

    ### ADDING SPECIES: add has_ parameter and add condition to append to tuple.
    # In the init, all species that are present within the model should be set to True.
    # All couplings present between species should be set to true. 
    def __init__(self,
                 ellmin = 2,
                 ellmax = 2500,
                 lensing = False,
                 has_MasslessNeutrinos=False, 
                 has_MassiveNeutrinos=False,
                 bbn_type = "",
                 linx_reaction_net = "key_PRIMAT_2023"
                 ): 

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
        self.PArthENoPE_CLASS_table = jnp.asarray(np.loadtxt(file_dir+'/sBBN_2025_CLASS.txt'))
        self.bbn_type = bbn_type
        self.linx_reaction_net = linx_reaction_net
    
    # @jit
    @eqx.filter_jit
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
        params = self.add_derived_parameters(params)
        PT, BG = self.get_PTBG(params)
        #return self.SS.Pk_lin(PT.k, 0., PT, BG)
        #return PT.delta_b
        Cls = self.SS.get_Cl(PT, BG, params)
        return Cls

    # @jit
    @eqx.filter_jit
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
        PE = perturbations.PerturbationEvolver(self.perturbations_list, BG, params)
        
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

        if self.bbn_type=="Table" or self.bbn_type=="table":
            # interpolate CLASS ParthENoPE table
            bbn = self.PArthENoPE_CLASS_table
            omegab_all = bbn[:, 0]
            DNeff_all = bbn[:, 1]
            YHe_all = bbn[:, 2]

            unique_DNeff, idxNeff = jnp.unique(DNeff_all, return_index=True)
            n2 = unique_DNeff.size
            n1 = bbn.shape[0] // n2

            omegab = omegab_all[:n1]
            DNeff = unique_DNeff

            YHe_grid = YHe_all.reshape(n2, n1)
            
            # Neff = params["Neff"] # less extensible option
            a_bbn = cnst.TCMB_today*1e-6/0.01   # neutrino decoupling is well over by 10 keV, so 
                                                # compute Neff at a scale factor approximately 
                                                # corresponding to this temperature
            lna_bbn = jnp.log(a_bbn)

            # last two args are user input omega_b and (Neff_BBN - 3.046) (MUST be 3.046 as 
            # this was assumed when constructing the PArthENoPE table)

            # we want to loop through everything possibly contributing to Neff.  However,
            # we need to triple the contribution from neutrinos.  So we add extra terms for 
            # neutrino contributions
            Neff = (jnp.sum(jnp.asarray([s.rho(lna_bbn, params) for s in self.species_list])) + 2*self.species_list[-1].rho(lna_bbn,params) - 
                    self.species_list[-2].rho(lna_bbn,params))/self.species_list[-1].rho(lna_bbn,params)
            print(Neff)
            res_YHe = bilinear_interp(omegab, DNeff,YHe_grid, params['omega_b'],Neff - 3.046)
            params['YHe'] = res_YHe
            print(params['YHe'])

        elif self.bbn_type=="LINX" or self.bbn_type=="Linx" or self.bbn_type=="linx":
            if params.get("Neff") is not None:
                print("You have specified a value of Neff, but LINX expects a parameter " \
                    "dNnu which will be used to compute Neff.  Refer to LINX docs or " \
                    "https://arxiv.org/abs/2408.14538 for more information.")
                sys.exit()

            thermo_model_DNeff = BackgroundModel()
            (
                t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, Neff_vec 
            ) = thermo_model_DNeff(jnp.asarray(params['dNnu']))

            abundance_model = AbundanceModel(NuclearRates(nuclear_net=self.linx_reaction_net))

            abundances = abundance_model(
                rho_g_vec,
                rho_nu_vec,
                rho_NP_vec,
                P_NP_vec,
                t_vec=t_vec_ref,
                a_vec=a_vec_ref,     # CG add eta fac, tau n fac, nuclear_rates_q
                )
            
            YHe_BBN = 4*abundances[5]
            params['Neff'] = Neff_vec[-1]
            params['YHe'] = YHe_BBN # CG NEED TO CONVERT TO WHAT ABCMB EXECTS
            # run LINX
        
        params['N_ur']         = params['Neff'] - (params['T_ncdm'] / params['TCMB0'])**4 / (4. / 11.)**(4. / 3.) * params['N_ncdm']
        params['omega_nu']     = 7. / 8. * params['N_ur'] * (4. / 11.)**(4. / 3.) * params['omega_g']
        params['omega_r']      = params['omega_g'] + params['omega_nu']
        params['R_nu']         = jnp.where(params['omega_r'] > 0.0, params['omega_nu'] / params['omega_r'], 0.0)
        params['omega_Lambda'] = params['h']**2 - params['omega_r'] - params['omega_m']

        return params