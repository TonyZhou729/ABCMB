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
from .utils.deprecation import deprecated
from .utils.batching import stack_batch, jvp_stage

from .linx.background import BackgroundModel
from .linx.abundances import AbundanceModel
from .linx.nuclear import NuclearRates
from . import model_helpers

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
    run_one_cosmology : Compute CMB angular and matter power spectra for one cosmology
    run_parallel_cosmologies : Compute the same spectra for several cosmologies at once (batched)
    run_parallel_cosmologies_jvp : Compute batched TT, TE, and EE spectra, with forward-mode derivatives
    __call__ : Deprecated alias of run_one_cosmology, to be removed in 1.2.0
    add_derived_parameters : Compute derived parameters
    run_cosmology_abbr : Run the pipeline on parameters that already include derived keys
    get_BG_pre_recomb : Get pre-recombination background and HyRex inputs
    run_post_recomb : Run the post-recombination stage (background, perturbations, spectra)
    get_PTBG : Get perturbation table and background cosmology
    get_BG : Get background cosmology
    get_BG_batched : Get background cosmology for several cosmologies at once, optionally with tangents
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

    def run_parallel_cosmologies(self, params_list, k_chunk_size=100, shard=None):
        """
        Batched evaluation of several cosmologies at once (uses a refactored pipeline
        for performance).

        Parameters:
        -----------
        params_list : list of dict
            Cosmological parameters, one dict per cosmology
        k_chunk_size : int, optional
            Number of wavenumbers per compiled perturbation kernel.  Defaults
            to 100.
        shard : bool or None, optional
            Split cosmologies across visible GPUs (fails if no GPUs).  This will pad batches 
            to a multiple of the number of GPUs by repeating the last cosmology (removed from 
            output). True always attempts to shard, False restricts to one device, None shards 
            if multiple devices are automatically detected.  Defaults to None.

        Returns:
        --------
        Output
            Spectra and parameters with a leading batch axis; BG and PT are None

        """
        num_cosmologies = len(params_list)
        params, _, shardfn, _, _ = stack_batch([self.add_derived_parameters(p) for p in params_list], shard=shard)
        BG, _ = self.get_BG_batched(params, shardfn=shardfn)
        
        PT = self.PE.full_evolution_batched((BG, params), k_chunk_size=k_chunk_size)

        # params may have been augmented by stack_batch, so trim x here to num_cosmologies
        Cls, Pk, Pk_nl, params = jax.tree_util.tree_map(lambda x: x[:num_cosmologies], self._spectra_vmapped(PT, BG, params) + (params,))
        lensed = self.specs["lensing"]
        return Output(Cls[0], Cls[1], Cls[2], Cls[3] if lensed else None, Cls[4] if lensed else None,
                      Pk, self.SS.ells, self.SS.k_axis_Pk_output, None, None, params, Pk_nl=Pk_nl)

    def run_parallel_cosmologies_jvp(self, params_list, params_dots, k_chunk_size=100, shard=None, spectra_chunk=None):
        """
        Batched TT, TE and EE and their forward-mode derivatives along several directions.

        Runs the stages of run_parallel_cosmologies one at a time under jax.jvp, with
        all directions pushed through each stage together, so the pipeline is solved
        once per call and keeps run_parallel_cosmologies' batching, k-chunking and
        sharding.

        Parameters:
        -----------
        params_list : list of dict
            Cosmological parameters including the derived ones (add_derived_parameters),
            one dict per cosmology
        params_dots : list of dict
            One dict per direction: tangents of entries of params_list, each a scalar
            or an array with a leading axis over the cosmologies; missing entries are
            zero. For a direction in the input parameters, take jax.jvp of
            add_derived_parameters for each cosmology.
        k_chunk_size : int, optional
            Number of wavenumbers per compiled perturbation kernel.  Defaults to 100.
        shard : bool or None, optional
            As in run_parallel_cosmologies.  Defaults to None.
        spectra_chunk : int or None, optional
            Cosmologies per device in each call of the spectrum stage, which sets the
            memory peak.  Defaults to None (all at once).

        Returns:
        --------
        tuple
            (Cls, Cls_dot): Cls = (TT, TE, EE), each of shape (n_cosmologies, n_l);
            Cls_dot the same with a leading direction axis
        """
        params, params_dot, shardfn, dir_shardfn, n_dev = stack_batch(params_list, params_dots, shard)
        BG, BG_dot = self.get_BG_batched(params, params_dot, shardfn, dir_shardfn)
        PT, PT_dot = jvp_stage(lambda bg, p: self.PE.full_evolution_batched((bg, p), k_chunk_size=k_chunk_size),
                                (BG, params), (BG_dot, params_dot))

        num_padded = next(iter(params.values())).shape[0]
        step = num_padded if spectra_chunk is None else spectra_chunk * (n_dev if shardfn is not None else 1)
        parts = []
        for s in range(0, num_padded, step):
            prim = jax.tree_util.tree_map(lambda x: x[s:s + step] if eqx.is_array(x) and x.ndim > 0 else x, (PT, BG, params))
            tang = jax.tree_util.tree_map(lambda x: x[:, s:s + step] if eqx.is_array(x) and x.ndim > 1 else x,
                                          (PT_dot, BG_dot, params_dot))
            if shardfn is not None:
                prim, tang = shardfn(prim), dir_shardfn(tang)
            parts.append(jvp_stage(self._cls_vmapped, prim, tang))
        Cls = tuple(jnp.concatenate([c[i] for c, _ in parts])[:num_cosmologies] for i in range(3))
        Cls_dot = tuple(jnp.concatenate([d[i] for _, d in parts], axis=1)[:, :num_cosmologies] for i in range(3))
        return Cls, Cls_dot

    run_cosmology_abbr     = model_helpers.run_cosmology_abbr
    get_BG_pre_recomb      = model_helpers.get_BG_pre_recomb
    run_post_recomb       = model_helpers.run_post_recomb
    get_PTBG               = model_helpers.get_PTBG
    get_BG                 = model_helpers.get_BG
    add_derived_parameters = model_helpers.add_derived_parameters

    get_BG_batched             = model_helpers.get_BG_batched
    _get_BG_pre_recomb_vmapped = model_helpers._get_BG_pre_recomb_vmapped
    _get_BG_vmapped            = model_helpers._get_BG_vmapped
    _spectra_vmapped           = model_helpers._spectra_vmapped
    _cls_vmapped               = model_helpers._cls_vmapped

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
