import os
import sys

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import config, lax
from jaxtyping import Array

from . import background, model_specs, perturbations, spectrum
from . import constants as cnst
from .ABCMBTools import bilinear_interp
from .background import (
    Background,
    BackgroundPreRecomb,
    ReionizationModelFromTau,
    ReionizationModelFromZ,
)
from .hyrex import hyrex
from .linx import const as linxconst
from .linx import thermo as linxThermo
from .linx.abundances import AbundanceModel
from .linx.background import BackgroundModel
from .linx.nuclear import NuclearRates

file_dir = os.path.dirname(__file__)

config.update("jax_enable_x64", True)


class Model(eqx.Module):
    """
    Model configuration and computation manager.

    Creates instances of fluid species based on user input and organizes
    them for computation. Manages the full pipeline from background
    evolution through CMB power spectrum computation.

    Attributes:
    -----------
    PE : perturbations.PerturbationEvolver
        ABCMB perturbations module
    SS : spectrum.SpectrumSolver
        ABCMB spectrum module
    RecModel : hyrex.recomb_model
        HyRex recombination module
    specs : dict
        A dictionary of run options (expected to be static)
    species_list : tuple 
        A list of all fluids in the user cosmology
    species_dict : dict 
        A dictionary containing the names of all fluids, in the same order as 
        they appear in species_list.
    PArthENoPE_CLASS_table  : Array 
        A 2D table for interpolation of the helium-4 mass fraction based
        on the user's input baryon density and Neff
    thermo_model_DNeff : linx.BackgroundModel
        A LINX background model for BBN thermodynamics
    abundanceModel : linx.AbundanceModel
        A LINX abundance model used for computing the helium-4 mass fraction
        given the user's input baryon density, Neff, neutron lifetime, and
        nuclear reaction rates.
    adjoint : diffrax.adjoint
        Adjoint mode for diffrax solves.  Default is ForwardMode.

    Methods:
    --------
    __call__ : Compute CMB angular power spectra
    get_PTBG : Get perturbation table and background cosmology
    get_BG : Get background cosmology
    add_derived_parameters : Compute derived parameters
    """

    PE : perturbations.PerturbationEvolver
    SS : spectrum.SpectrumSolver
    RecModel : hyrex.recomb_model
    specs : dict

    species_list : tuple = ()
    species_dict : dict

    PArthENoPE_CLASS_table  : Array
    thermo_model_DNeff : BackgroundModel
    abundanceModel : AbundanceModel

    adjoint : "diffrax.adjoint" = eqx.field(static=True)

    ### ADDING SPECIES: add has_ parameter and add condition to append to tuple.
    # In the init, all species that are present within the model should be set to True.
    # All couplings present between species should be set to true. 
    def __init__(self,
                 user_species=None,
                 **kwargs
                 ):
        """
        Initialize Model instance.

        Sets up fluid species, recombination model, and spectrum solver
        based on configuration parameters.

        Parameters:
        -----------
        user_species : tuple
            A tuple of user-defined fluids to be included in the cosmology
        **kwargs : dict
            Configuration options passed as keyword arguments.
            Any unknown keys will be preserved for custom species extensibility.
        """

        # Pull adjoint out of kwargs before load_specs — it must NOT end up
        # inside self.specs (a non-JAX pytree leaf breaks lax.cond / filter_jit
        # tracing).
        adjoint = kwargs.pop("adjoint", diffrax.ForwardMode)

        # Fill in all user defined and missing specs parameters
        specs = model_specs.load_specs(kwargs)
        self.specs = specs

        # Populate all species
        self.species_list, self.species_dict = model_specs.populate_species(
            user_species,
            specs,
        )   

        # Initialize perturbation evolver
        k_axis_perturbations, k_axis_Pk_output = model_specs.get_k_axis_perturbations(specs)
        self.PE = perturbations.PerturbationEvolver(
            self.species_list,
            self.species_dict,
            k_axis_perturbations,
            specs,
            adjoint=adjoint,
        )

        # Intialize spectrum solver
        k_axis_transfer = model_specs.get_k_axis_transfer(specs)
        self.SS = spectrum.SpectrumSolver(
            specs["l_min"],
            specs["l_max"],
            specs["lensing"],
            k_axis_transfer,
            k_axis_Pk_output,
            k_pivot=specs["k_pivot"],
            scale_sw=specs["scale_sw"],
            scale_isw=specs["scale_isw"],
            scale_dop=specs["scale_dop"],
            scale_pol=specs["scale_pol"]
        )

        # Initialize recombination model.
        self.RecModel = hyrex.recomb_model(adjoint=adjoint) # DO NOT CHANGE z1 FROM 0

        # Initialize BBN model
        self.PArthENoPE_CLASS_table = jnp.asarray(np.loadtxt(file_dir+'/sBBN_2025_CLASS.txt'))
        # initialize LINX
        if self.specs["bbn_type"].lower() == "linx":
            self.thermo_model_DNeff = BackgroundModel(adjoint=adjoint)
            self.abundanceModel = AbundanceModel(NuclearRates(nuclear_net=self.specs["linx_reaction_net"]), adjoint=adjoint)
        else:
            self.thermo_model_DNeff = None
            self.abundanceModel = None

        self.adjoint = adjoint

    # need this outside of the main jit context
    # since we want LINX/HyRex to run on CPU
    def __call__(self, params : dict = {}):
        """
        Runs the full pipeline from background evolution through
        perturbation integration to CMB power spectrum computation.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        Output
            Bundle of CMB power spectra (ClTT, ClTE, ClEE) and their
            multipole grid l, matter power spectrum Pk and its k-grid,
            the Background and PerturbationTable objects, and the
            full parameter dict including derived keys.
        """
        full_params = self.add_derived_parameters(params)
        return self.run_cosmology_abbr(full_params)


    def run_cosmology_abbr(self, params : dict):
        """
        Compute CMB angular power spectra for given parameters.

        Runs the full pipeline from background evolution through
        perturbation integration to CMB power spectrum computation.

        Parameters:
        -----------
        params : dict
            Cosmological parameters (must already have derived keys).

        Returns:
        --------
        Output
            CMB power spectra and friends.
        """
        # Cast int/bool params to float64 before entering any
        # ``eqx.filter_jit`` for custom_vjp/AD safety in 
        # checkpointed_while_loop
        def _to_float(v):
            arr = jnp.asarray(v)
            if arr.dtype.kind in 'iub':
                return arr.astype(jnp.float64)
            return arr
        params = jax.tree_util.tree_map(_to_float, params)

        pre_BG = self.get_BG_pre_recomb(params)

        cpu_dev = jax.devices('cpu')[0]
        recomb_inputs_cpu = jax.device_put(pre_BG.recomb_inputs, cpu_dev)
        params_cpu = jax.device_put(params, cpu_dev)

        recomb_output = eqx.filter_jit(self.RecModel, backend='cpu')((recomb_inputs_cpu, params_cpu))

        try:
            recomb_output = jax.device_put(recomb_output, jax.devices('gpu')[0])
        except Exception:
            pass

        # recomb_output contains array_with_padding objects whose
        # padding_size and lastnum int arrays.  The
        # checkpointed_while_loop's filter_custom_vjp inside
        # _run_post_recomb's diffrax solves trips an internal
        # _get_value_assert_unperturbed on int leaves under outer
        # AD; convert to float to avoid.
        recomb_output = jax.tree_util.tree_map(_to_float, recomb_output)

        return self._run_post_recomb(params, pre_BG, recomb_output)

    @eqx.filter_jit
    def get_BG_pre_recomb(self, params : dict):
        """
        Pre-recomb stage: tabulate conformal time and bundle H, T, nH for recombination.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        BackgroundPreRecomb
        """
        # let the user know the code is compiling
        print("")
        print('              /\\  ')
        print('             /  \\   ')
        print('            / /\\ \\  ')
        print('           / /__\\ \\    ___   ___  ')
        print('          / ______ \\  | _ \\ / __\\ _  _  ')
        print('         / /      \\ \\ |  _// /   | \\/ | __  ')
        print('        / /        \\ \\| _ \\\\ \\___||\\/||| -)  ')
        print('       /_/          \\_|___/ \\___/||  |||_-) is compiling...')
        print('\\_____/      ')
        print("")
        return BackgroundPreRecomb(params, self.species_list, self.RecModel, adjoint=self.adjoint)

    @eqx.filter_jit
    def _run_post_recomb(self, params : dict, pre_BG : "BackgroundPreRecomb", recomb_output):
        """
        Post-recombination stage: full Background construction (reionization,
        optical depth, decoupling), perturbation evolution, CMB spectra.

        Parameters:
        -----------
        params : dict
            Cosmological parameters
        pre_BG : BackgroundPreRecomb
            Output of :meth:`get_BG_pre_recomb`.
        recomb_output : tuple
            HyRex output ``(xe, lna_xe, Tm, lna_Tm)``.

        Returns:
        --------
        Output
        """

        # Compute background and linear perturbations
        PT, BG = self.get_PTBG(params, pre_BG, recomb_output)

        # Compute CMB power spectra
        Cls = self.SS.get_Cl(PT, BG, params)
        l = self.SS.ells

        # Compute linear matter power spectrum
        Pk = self.SS.Pk_lin(self.SS.k_axis_Pk_output, 0., PT, params)
        k = self.SS.k_axis_Pk_output

        # Package
        output = Output(
            Cls[0], Cls[1], Cls[2], Pk,
            l, k, BG, PT, params
        )

        return output

    @eqx.filter_jit
    def get_PTBG(self, params : dict, pre_BG : "BackgroundPreRecomb", recomb_output):
        """
        Get perturbation table and full Background.

        Constructs the post-recomb Background from ``pre_BG`` + ``recomb_output``
        and runs the perturbation evolver.

        Parameters:
        -----------
        params : dict
            Cosmological parameters
        pre_BG : BackgroundPreRecomb
            Pre-recombination stage object.
        recomb_output : tuple
            HyRex output ``(xe, lna_xe, Tm, lna_Tm)``.

        Returns:
        --------
        tuple
            (PerturbationTable, Background)
        """
        BG = self.get_BG(params, pre_BG, recomb_output)
        PT = self.PE.full_evolution((BG, params))
        return PT, BG

    def get_BG(self, params : dict, pre_BG : "BackgroundPreRecomb", recomb_output):
        """
        Construct the full ``Background`` from pre-recomb + HyRex output.

        Selects the reionization model (z-input vs tau-input) via ``lax.cond``.
        NOT directly ``@eqx.filter_jit``-decorated; called from inside
        ``_run_post_recomb`` (which is jit-wrapped).

        Parameters:
        -----------
        params : dict
            Cosmological parameters
        pre_BG : BackgroundPreRecomb
            Pre-recombination stage object.
        recomb_output : tuple
            HyRex output ``(xe, lna_xe, Tm, lna_Tm)``.

        Returns:
        --------
        background.Background
        """
        def get_BG_z_reion(args):
            params, pre_BG, recomb_output = args
            return Background(pre_BG, recomb_output, params, ReionizationModelFromZ)

        def get_BG_tau_reion(args):
            params, pre_BG, recomb_output = args
            return Background(pre_BG, recomb_output, params, ReionizationModelFromTau)

        BG = lax.cond(
            self.specs["input_tau_reion"],
            get_BG_tau_reion,
            get_BG_z_reion,
            (params, pre_BG, recomb_output)
        )

        return BG

    def add_derived_parameters(self, param_in : dict) -> dict:
        # we do not want to do in-place updates so we can
        # recycle dicts if LINX option is used
        params = param_in.copy()

        # Default parameters except Neff and YHe
        params['h']             = jnp.array(params.get('h', 0.6736))
        params['H0']            = jnp.array(params['h'] * cnst.H0_over_h)
        params['omega_cdm']     = jnp.array(params.get('omega_cdm', 0.120))
        params['omega_b']       = jnp.array(params.get("omega_b", 0.02237))
        params['A_s']           = jnp.array(params.get('A_s', 2.1e-9))
        params['n_s']           = jnp.array(params.get('n_s', 0.9649))
        params['TCMB0']         = jnp.array(params.get('TCMB0', 2.34865418e-4))

        # Reionization
        if self.specs["input_tau_reion"]:
            params['tau_reion'] = jnp.array(params.get('tau_reion', 0.0544))
        else:
            params['z_reion'] = jnp.array(params.get('z_reion', 7.67))
        params['Delta_z_reion'] = jnp.array(params.get('Delta_z_reion', 0.5))
        params['z_reion_He']    = jnp.array(params.get('z_reion_He', 3.5))
        params['Delta_z_reion_He'] = jnp.array(params.get('Delta_z_reion_He', 0.5))
        params['exp_reion']     = jnp.array(params.get('exp_reion',1.5))

        # Here we fill in a fake omega_Lambda just so that the DE energy density can be computed in a loop.
        # This fake quantity will not be used in anything, and later the correct omega_Lambda will be computed.
        # Purely computational, no physics used or messed up.
        params['omega_Lambda'] = 0.

        # Massive neutrinos
        params['T_nu_massive']  = jnp.array(params.get('T_nu_massive', 0.71611)) # Massive neutrino temperature, as a ratio to TCMB.
        params['N_nu_massive']  = jnp.array(params.get('N_nu_massive', 0))  # Number of massive neutrinos
        params['m_nu_massive']  = jnp.array(params.get('m_nu_massive', 0.06)) # Massive neutrino mass, in eV

        ### CHECKING INPUT COMPATIBILITY ###

        input_N    = params.get('N_nu_massless') is not None
        input_Neff = params.get('Neff') is not None
        input_T_nu_massless = params.get('T_nu_massless') is not None

        # If the user input both massless neutrino number and Neff, throw an error. Our code treats these as 1-to-1, see paper.
        if input_N and input_Neff:
            print("You can only input one of N_nu_massless or Neff, but got values N_nu_massless={} and Neff={}.".format(params["N_nu_massless"], params["Neff"]))
            sys.exit()

        # If the user input either N_massless or Neff, but requested LINX, throw an error. LINX will compute the correct values.
        if (input_N or input_Neff or input_T_nu_massless) and self.specs["bbn_type"].lower() == "linx":
            print(
                "You have specified a value for N_nu_massless and/or Neff and/or T_nu_massless, \n"
                "but LINX instead expects a parameter 'Delta_Neff_init' which will be used to \n" \
                "compute Neff. Refer to LINX docs or https://arxiv.org/abs/2408.14538 for more info.\n" \
            )
            sys.exit()

        if not input_N and not input_Neff and self.specs["bbn_type"].lower() != "linx":
            params["N_nu_massless"] = 3 - params['N_nu_massive']
            input_N = True

        ### END OF INPUT COMPATIBILITY ###
        # now that we have verified the user put in the right parameters we can set T_nu_massless
        params['T_nu_massless'] = jnp.array(params.get('T_nu_massless', 0.71636856)) # Massless neutrino temperature, as a ratio to TCMB

        ### HELIUM FRACTION AND Neff ###
        # Regardless of bbn_type, these two parameters will be set by the end.

        lna_early = -23.
        a_early = jnp.exp(lna_early)

        # Case 1: The user specifies the true number of massless neutrinos. Note this is distinct from CLASS' N_ur which
        # is computed assuming T_massless = (4/11)^(1/3) x T_CMB.
        # Here, Neff will be inferred from the cosmological fluid content. 
        # In particular if the universe contains massive neutrinos, we account for the error incurred when using a late time
        # massive neutrino temperature which underestimates the massive neutrino energy density at early time, when Neff is set.
        # We account for this by adding the missing relativistic energy in massive neutrinos at early times to the massless fluid.
        # See detail in paper.
        if input_N:
            rho_g = 0.
            rho_nu = 0.
            rho_extra = 0.
            for s in self.species_list:
                rho = s.rho(lna_early, params)
                if s.name == "Photon":
                    rho_g += rho
                elif "neutrino" in s.name.lower():
                    rho_nu += rho
                else:
                    rho_extra += rho

            Neff_raw     = (rho_nu+rho_extra)/rho_g * (8./7.) * (11./4.)**(4./3.) # Uncorrected Neff using T_nu_massive today
            rho_nu_early = 7/8 * (params["N_nu_massless"] + params["N_nu_massive"]) * params["T_nu_massless"]**4 * rho_g # Correct using massless neutrino temp.
            params["Neff"] = (rho_nu_early+rho_extra)/rho_g * (8./7.) * (11./4.)**(4./3.)
            params["N_nu_massless"] = params["N_nu_massless"] + params["Neff"] - Neff_raw # Add difference to massless sector.

        if self.specs["bbn_type"].lower() == "table":
            # Applies if user requested BBN table to be user.
            # In this case Neff must already have been set, and can be used to interp YHe.

            # interpolate CLASS ParthENoPE table
            bbn = self.PArthENoPE_CLASS_table
            omegab_all = bbn[:, 0]
            DNeff_all = bbn[:, 1]
            YHe_all = bbn[:, 2]

            # we have to hardcode these values to be jit safe (alternatively we 
            # could read them in at runtime, but these tables don't update 
            # frequently)
            n2 = 13 
            n1 = 701

            omegab = omegab_all[:n1]
            DNeff = DNeff_all[::n1] 

            YHe_grid = YHe_all.reshape(n2, n1)
            
            # Neff = params["Neff"] # less extensible option

            # Comprehensive Neff, includes all relativsitic species at early times.
            Neff_BBN = params["Neff"]
            
            # last two args are user input omega_b and (Neff_BBN - 3.046) (MUST be 3.046 as 
            # this was assumed when constructing the PArthENoPE table)
            res_YHe = bilinear_interp(omegab, DNeff,YHe_grid, params['omega_b'],Neff_BBN - 3.046)

            # tabulated result is Yp_CMB
            params['YHe'] = res_YHe
        elif self.specs["bbn_type"].lower() == "linx":
            # Applies if user requested to run LINX.
            # For this branch to happen, Neff must NOT have already been set. 
            # Logic above has already accounted for this, since input_T and input_Neff must both be False
            # for LINX to execute.
            params['Delta_Neff_init'] = jnp.array(params.get('Delta_Neff_init', 0.))
            (
                t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, Neff_vec 
            ) = eqx.filter_jit(self.thermo_model_DNeff,backend='cpu')(params['Delta_Neff_init'])

            # convert user input omega_b to eta_fac LINX expects
            eta_fac = params['omega_b'] * linxconst.Omegabh2_to_eta0/linxconst.eta0

            abundances = eqx.filter_jit(self.abundanceModel,backend='cpu')(
                rho_g_vec,
                rho_nu_vec,
                rho_NP_vec,
                P_NP_vec,
                t_vec=t_vec_ref,
                a_vec=a_vec_ref,  
                eta_fac = eta_fac,
                tau_n_fac = jnp.asarray(params.get("tau_n_fac", 1.0)),
                nuclear_rates_q = jnp.asarray( params.get("nuclear_rates_q", jnp.zeros( len(self.abundanceModel.nuclear_net.reactions) )) )
                )
  
            # number abundance
            try:
                params['T_nu_massless'] = jax.device_put(
                    linxThermo.T_nu(rho_nu_vec[-1]) / linxThermo.T_g(rho_g_vec[-1]),
                    device=jax.devices('gpu')[0]
                )
                params['Neff'] = jax.device_put(Neff_vec[-1],device=jax.devices('gpu')[0])
                YHe_BBN = jax.device_put(4*abundances[5],device=jax.devices('gpu')[0])
            except Exception: # no GPU
                params['T_nu_massless'] = linxThermo.T_nu(rho_nu_vec[-1]) / linxThermo.T_g(rho_g_vec[-1])
                params['Neff'] = Neff_vec[-1]
                YHe_BBN = 4*abundances[5]
                pass
        
            # CMB uses real mass fraction
            Yp_CMB = 1./(4*cnst.mH/cnst.mHe*(1/YHe_BBN - 1) + 1)
            params['YHe'] = Yp_CMB

            # Now Neff has been set by LINX but massless neutrino number has yet to be calculated.
            # we now set the input_Neff flag to True so the branch below takes care of this.
            input_Neff = True
        else:
            # Applies if user wanted neither LINX or BBN table. 
            params['YHe'] = jnp.array(params.get('YHe', 0.245))


        # Case 2: User specifies the total Neff of the universe, including neutrinos and all other relativistic species at early times.
        # Then we subtract off all relativistic energy densities from Neff and assign the remaining to massless neutrinos.
        # Since massless neutrino temperature is already specified here, the true derived parameter is N_nu_massless, the physical
        # number of massless neutrinos.
        # The philosophy is that if we're increasing Neff, we are not heating the existing neutrinos, we are adding extra neutrinos
        # at the same temperature. At the CMB level these are indistinguishable, but we chose the later convention. 
        # Note, if after the deduction there's not enough energy density for massless neutrinos (N_nu_massless < 0), ABCMB throws an error.
        if input_Neff:
            rho_g = 0.
            rho_extra = 0.
            for s in self.species_list:
                if s.name == "Photon":
                    rho = s.rho(lna_early, params)
                    rho_g += rho
                elif s.name != "MasslessNeutrino":
                    rho = s.rho(lna_early, params)
                    rho_extra += rho
            rho1nu = 7/8 * (4/11)**(4/3) * rho_g
            
            params['N_nu_massless'] = (params["Neff"] - rho_extra/rho1nu) * ((4/11)**(1/3) / params["T_nu_massless"])**4

        # Loop over matter fluids to compute total matter density today.
        rho_m = 0.
        for s in self.species_list:
            if s.is_matter:
                rho_m += s.rho(0., params)
        params['omega_m']      = rho_m / (3 * cnst.H0_over_h**2/8/jnp.pi/cnst.G) # Fractional matter density
        params['R_b']          = params['omega_b'] / params['omega_m'] # Baryon fraction
    
        # Loop over all fluids and compute energy density at very early time, inferring radiation energy density this way.
        a_early = jnp.exp(-23.)
        rho_r  = 0.
        rho_nu = 0.
        for s in self.species_list:
            rho_r += s.rho(jnp.log(a_early), params)
            if "neutrino" in s.name.lower():
                rho_nu += s.rho(jnp.log(a_early), params)

        params['omega_r']      = rho_r * a_early**4 / (3 * cnst.H0_over_h**2/8/jnp.pi/cnst.G) # Fractional radiation density today
        params['R_nu']         = rho_nu / rho_r # Fractional radiation density in neutrinos, defined at early times. Used for setting adiabatic ICs.

        # Special density parameter defined for computing adiabatic initial conditions
        # Defined as Omega_m / sqrt{Omega_r} * H0, in units of 1/Mpc
        params['om'] = params['omega_m'] / jnp.sqrt(params['omega_r']) * cnst.H0_over_h / cnst.c_Mpc_over_s

        # Having inferred correct omega_m and omega_r, compute correct omega_Lambda
        params['omega_Lambda'] = params['h']**2 - params['omega_r'] - params['omega_m']

        # There is NO NEED to modify this list!!  This is to make sure any new
        # user-defined keys will not trigger recompilation by wrapping them in
        # jnp.array, as is done manually above for all other keys.  LINX-
        # related inputs are intentionally excluded from this list!
        expected_keys = {
            'h', 'H0', 'omega_cdm', 'omega_b', 'A_s', 'n_s', 'TCMB0',
            'tau_reion', 'z_reion', 'Delta_z_reion', 'z_reion_He', 'Delta_z_reion_He', 'exp_reion',
            'omega_Lambda', 'T_nu_massive', 'N_nu_massive', 'm_nu_massive',
            'N_nu_massless', 'Neff', 'T_nu_massless', 'YHe',
            'omega_m', 'R_b', 'omega_r', 'R_nu', 'om'
        }
        
        for key, value in param_in.items():
            if key not in expected_keys:
                params[key] = jnp.array(value)

        return params

class Output(eqx.Module):
    """
    Object containing final and intermediate results from one cosmological simulation.

    Attributes:
    -----------
    ClTT : jnp.array
        Temperature-temperature power spectrum
    ClTE : jnp.array
        Temperature-polarization power spectrum
    ClEE : jnp.array
        Polarization-polarization power spectrum
    Pk : jnp.array
        Matter power spectrum
    l : jnp.array
        Multipoles l at which ClTT/ClTE/ClEE are output
    k : jnp.array
        Wavenumbers k at with Pk is output
    BG  : background.Background
        Background object containing functions like Hubble, recombination history, etc
    PT : perturbations.PerturbationTable
        Perturbation table including perturbations for all fluids
    params : dict
        Complete parameter dictionary including derived parameters
    """

    # Power spectra
    ClTT : jnp.array
    ClTE : jnp.array
    ClEE : jnp.array
    Pk   : jnp.array

    l  : jnp.array
    k  : jnp.array
    BG : background.Background
    PT : perturbations.PerturbationTable
    params : dict
