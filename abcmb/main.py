from jax import config
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np
import equinox as eqx

import diffrax
import jax

import os
file_dir = os.path.dirname(__file__)

from .hyrex import hyrex
from . import background, perturbations, spectrum, model_specs
from .background import BackgroundPreRecomb
from .utils.deprecation import deprecated

from .linx.background import BackgroundModel
from .linx.abundances import AbundanceModel
from .linx.nuclear import NuclearRates
from . import model_helpers

config.update("jax_enable_x64", True)

def _to_float(tree):
    """
    Cast int/bool leaves to float64.

    Parameters:
    -----------
    tree: pytree
        Pytree whose leaves are to be converted to float.
    
    Returns:
    --------
    array
        Array of leaves as floats
        
    """
    def cast(v):
        arr = jnp.asarray(v)
        return arr.astype(jnp.float64) if arr.dtype.kind in 'iub' else arr

    return jax.tree_util.tree_map(cast, tree)


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
    run_one_cosmology : Compute CMB angular and matter power spectra for one cosmology
    __call__ : Deprecated alias of run_one_cosmology, to be removed in 1.2.0
    add_derived_parameters : Compute derived parameters
    run_cosmology_abbr : Run the pipeline on parameters that already include derived keys
    get_BG_pre_recomb : Get pre-recombination background and HyRex inputs
    run_post_recomb : Run the post-recombination stage (background, perturbations, spectra)
    get_PTBG : Get perturbation table and background cosmology
    get_BG : Get background cosmology

    All methods other than ``__init__``, ``__call__`` and ``run_one_cosmology``
    are defined in ``abcmb.model_helpers`` and attached to the class by assignment.
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
            scale_pol=specs["scale_pol"],
            use_bessel_tables=specs["use_bessel_tables"],
            delta_l_max=specs["delta_l_max"],
            n_k_cmb=specs["k_size_cmb"],
            nonlinear=specs["nonlinear"],
            halofit_prescription=specs["halofit_prescription"],
            # use only k that were computed for Pk_lin
            halofit_k_min=float(k_axis_perturbations[0]),
            halofit_k_max=float(k_axis_perturbations[-1]),
            halofit_k_per_decade=specs["halofit_k_per_decade"],
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
    @deprecated(since="1.0.0", removed_in="1.2.0", alternative="Model.run_one_cosmology")
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
            multipole grid l, linear matter power spectrum Pk (and the
            non-linear Pk_nl when ``nonlinear="halofit"``) and its k-grid,
            the Background and PerturbationTable objects, and the
            full parameter dict including derived keys.
        """
        full_params = self.add_derived_parameters(params)
        return self.run_cosmology_abbr(full_params)

    # need this outside of the main jit context
    # since we want LINX/HyRex to run on CPU
    def run_one_cosmology(self, params : dict = {}):
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
            multipole grid l, linear matter power spectrum Pk (and the
            non-linear Pk_nl when ``nonlinear="halofit"``) and its k-grid,
            the Background and PerturbationTable objects, and the
            full parameter dict including derived keys.
        """
        full_params = self.add_derived_parameters(params)
        return self.run_cosmology_abbr(full_params)

    run_cosmology_abbr     = model_helpers.run_cosmology_abbr
    get_BG_pre_recomb      = model_helpers.get_BG_pre_recomb
    run_post_recomb       = model_helpers.run_post_recomb
    get_PTBG               = model_helpers.get_PTBG
    get_BG                 = model_helpers.get_BG
    add_derived_parameters = model_helpers.add_derived_parameters

    def call_batched(self, params_list, k_chunk_size=100):
        """
        Batched evaluation of several cosmologies at once (uses a refactored pipeline
        for performance).

        Parameters:
        -----------
        params_list : list of dict
            Cosmological parameters, one dict per cosmology
        k_chunk_size : int
            Number of wavenumbers per compiled perturbation kernel.  Defaults
            to 100.

        Returns:
        --------
        Output
            Spectra and parameters with a leading batch axis; BG and PT are None

        """
        params, BG = self.get_BG_batched([self.add_derived_parameters(p) for p in params_list])
        PT = self.PE.full_evolution_batched((BG, params), k_chunk_size=k_chunk_size)
        Cls, Pk, Pk_nl = self._spectra_vmapped(PT, BG, params)
        lensed = self.specs["lensing"]
        return Output(Cls[0], Cls[1], Cls[2], Cls[3] if lensed else None, Cls[4] if lensed else None,
                      Pk, self.SS.ells, self.SS.k_axis_Pk_output, None, None, params, Pk_nl=Pk_nl)

    def get_BG_batched(self, params_list):
        """
        Batched evaluation of Backgrond module for a list of parameter dicts that
        already contain derived parameters.

        Parameters:
        -----------
        params_list : list of dict
            Cosmological parameters, one dict per cosmology

        Returns:
        --------
        tuple
            (params, BG), stacked along a leading batch axis
        """
        params = _to_float(jax.tree_utils.tree_map(lambda *xs: jnp.stack(xs), *params_list)) # need another tree map to stack the dicts to be converted
        pre_BG = self._get_BG_pre_recomb_vmapped(params)
        cpu_dev = jax.devices('cpu')[0]
        recomb_output =eqx.filter_jit(eqx.filter_vmap(self.RecModel), backup='cpu')(jax.device_put(pre_BG.recomb_inputs, cpu_dev), jax.device_put(params, cpu_dev))
        try:
            recomb_output = jax.device_put(recomb_output, jax.devices('gpu')[0])
        except Exception:
            pass
        return params, self._get_BG_vmapped(params, pre_BG, _to_float(recomb_output))

    @eqx.filter_jit
    def _get_BG_pre_recomb_vmapped(self, params):
        """
        get_BG_pre_recomb vmapped over the leading axis of params.

        Parameters:
        -----------
        params : array like of dict
            Cosmological parameters, one dict per cosmology

        Returns:
        --------
        array like
            Grid of pre-recomb BG results
        """
        return jax.vmap(lambda p: BackgroundPreRecomb(p, self.species_list, self.RecModel, adjoint=self.adjoint))(params)

    @eqx.filter_jit
    def _get_BG_vmapped(self, params, pre_BG, recomb_output):
        """
        get_BG vmapped over the leading axis of params.

        Parameters:
        -----------
        params : array like of dict
            Cosmological parameters, one dict per cosmology

        Returns:
        --------
        array like
            Grid of BG results
        """
        return jax.vmap(self.get_BG)(params, pre_BG, recomb_output)

    @eqx.filter_jit
    def _spectra_vmapped(self, PT, BG, params):
        """
        Cls vmapped over the leading axis of params.

        Parameters:
        -----------
        params : array like of dict
            Cosmological parameters, one dict per cosmology

        Returns:
        --------
        array like
            Grid of Cls
        """
        k = self.SS.k_axis_Pk_output
        Cls = jax.vmap(self.SS.get_Cl)(PT, BG, params)
        Pk = jax.vmap(self.SS.Pk_lin, in_axes=(None, None, 0, 0, 0))(k, 0., PT, BG, params)
        Pk_nl = None
        if self.SS.nonlinear == 'halofit':
            Pk_nl = jax.vmap(self.SS.Pk_nonlinear, in_axes=(None, None, 0, 0, 0))(k, 0., PT, BG, params)
        return Cls, Pk, Pk_nl

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
    Clpp : jnp.array or None
        Lensing potential power spectrum.  None if specs["lensing"]=False.
    ClBB : jnp.array or None
        Lensed B-mode power spectrum (induced from scalar perturbations).  None if specs["lensing"]=False.
    Pk : jnp.array
        Linear matter power spectrum at z=0
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
    Pk_nl : jnp.array or None
        HALOFIT non-linear matter power spectrum at z=0 on the same k-grid,
        or None if the model was built without ``nonlinear="halofit"``
    
    """

    # Power spectra
    ClTT : jnp.array
    ClTE : jnp.array
    ClEE : jnp.array
    Clpp : jnp.array
    ClBB : jnp.array
    Pk   : jnp.array

    l  : jnp.array
    k  : jnp.array
    BG : background.Background
    PT : perturbations.PerturbationTable
    params : dict

    Pk_nl : jnp.array = None
