import numpy as np
import jax.numpy as jnp
import equinox as eqx

from . import species

def load_specs(input_specs):

    specs = {}

    specs["use_LCDM_species"] = input_specs.get("use_LCDM_species", True)

    ### INPUT RELATED specs PARAMS ###
    # For reionization, input tau_reion the optical depth, or z_reion the hydrogen redshift?
    # WARNING: If the following parameter is set to True (by default) and the user inputs z_reion instead of tau_reion,
    # the default tau_reion will be used irrespective of the user input. The same is true if the user inputs tau_reion when
    # this is set to False. 
    specs["input_tau_reion"] = input_specs.get("input_tau_reion", True) 

    ### OUTPUT RELATED specs PARAMS ###
    specs["l_min"]     = input_specs.get("l_min", 2)
    specs["l_max"]     = input_specs.get("l_max", 2500)
    specs["lensing"]   = input_specs.get("lensing", False)
    specs["k_max"] = input_specs.get("k_max", 0.5)

    ### BESSEL FUNCTION OPTIONS ###
    specs["use_bessel_tables"] = input_specs.get("use_bessel_tables", True)

    ### BBN ###
    specs["bbn_type"] = input_specs.get("bbn_type", "")
    specs["linx_reaction_net"] = input_specs.get("linx_reaction_net", "key_PRIMAT_2023")

    ### Boltzmann Hierarchy Cutoffs ###
    specs["l_max_g"]     = input_specs.get("l_max_g", 12)
    specs["l_max_pol_g"] = input_specs.get("l_max_pol_g", 10)
    specs["l_max_massless_nu"]    = input_specs.get("l_max_massless_nu", 17)
    specs["l_max_massive_nu"]  = input_specs.get("l_max_massive_nu", 17)

    ### Perturbation k-grid resolution ###
    specs["k_step_sub"]             = input_specs.get("k_step_sub", 5.e-2)
    specs["k_step_super"]           = input_specs.get("k_step_super", 2.e-3)
    specs["k_step_transition"]      = input_specs.get("k_step_transition", 2.e-1)
    specs["k_step_super_reduction"] = input_specs.get("k_step_super_reduction", 1.e-1)
    specs["k_min_tau0"]             = input_specs.get("k_min_tau0", 1.e-1)
    specs["k_max_tau0_over_l_max"]  = input_specs.get("k_max_tau0_over_l_max", 1.8)
    # Extra multipoles computed beyond l_max so that the lensed spectra are
    # accurate up to l_max. Mirrors CLASS's precision parameter of the same
    # name (precisions.h, default 500), which CLASS folds into l_scalar_max
    # before deriving k_max (input.c: ppt->l_scalar_max += ppr->delta_l_max).
    specs["delta_l_max"]            = input_specs.get("delta_l_max", 500)
    # Sampling of the k-grid above the CMB ceiling, used only for the CMB lensing potential.
    specs["k_per_decade_for_pk"]    = input_specs.get("k_per_decade_for_pk", 10.)
    specs["k_per_decade_for_bao"]   = input_specs.get("k_per_decade_for_bao", 70.)
    specs["k_bao_center"]           = input_specs.get("k_bao_center", 3.)
    specs["k_bao_width"]            = input_specs.get("k_bao_width", 4.)
    # CLASS's "full Limber" k_max for C_l^phiphi. Set to 0 to disable the extension.
    specs["k_max_limber_over_l_max"] = input_specs.get("k_max_limber_over_l_max", 1.e-3)
    specs["H0_fid"]                 = input_specs.get("H0_fid", 2.255560e-04)
    specs["tau0_fid"]               = input_specs.get("tau0_fid",1.418668e+04)
    specs["rs_rec_fid"]             = input_specs.get("rs_rec_fid", 1.446279e+02)

    ### Transfer integration k-grid resolution ###
    specs["k_transfer_linstep"] = input_specs.get("k_transfer_linstep", 4.5e-1)
    specs["k_transfer_logstep"] = input_specs.get("k_transfer_logstep", 170.)
    specs["tau_rec_fid"]        = input_specs.get("tau_rec_fid", 281.040565)

    ### Pivot scale ###
    specs["k_pivot"]            = input_specs.get("k_pivot", 0.05)

    ### Set perturbations initial condition time ###
    specs["R_tc"] = input_specs.get("R_tc", 0.0015)
    specs["R_large"] = input_specs.get("R_large", 0.07)

    ### Perturbation Evolver Diffrax Settings ###
    specs["max_steps_PE"]    = input_specs.get("max_steps_PE", 2048)
    # Step size controller
    specs["k_split_PE"]      = input_specs.get("k_split_PE", 0.01)
    specs["rtol_small_k_PE"] = input_specs.get("rtol_small_k_PE", 1.e-5)
    specs["rtol_large_k_PE"] = input_specs.get("rtol_large_k_PE", 1.e-4)
    specs["atol_small_k_PE"] = input_specs.get("atol_small_k_PE", 1.e-10)
    specs["atol_large_k_PE"] = input_specs.get("atol_large_k_PE", 1.e-6)
    # Third tier, for the handful of modes added above the CMB ceiling purely for
    # the lensing potential. They only need phi+psi at late times, which is
    # smooth; resolving their photon oscillations to rtol_large_k_PE costs more
    # solver steps than every CMB mode combined.
    specs["rtol_limber_k_PE"] = input_specs.get("rtol_limber_k_PE", 1.e-3)
    specs["atol_limber_k_PE"] = input_specs.get("atol_limber_k_PE", 1.e-5)
    specs["pcoeff_PE"]       = input_specs.get("pcoeff_PE", 0.25)
    specs["icoeff_PE"]       = input_specs.get("icoeff_PE", 0.8)
    specs["dcoeff_PE"]       = input_specs.get("dcoeff_PE", 0.)

    ### Physical contributions to CMB temperature transfer function ###
    specs["scale_sw"]  = input_specs.get("scale_sw", 1)
    specs["scale_isw"] = input_specs.get("scale_isw", 1)
    specs["scale_dop"] = input_specs.get("scale_dop", 1)
    specs["scale_pol"] = input_specs.get("scale_pol", 1)

    # Preserve any unknown keys for custom species extensibility
    for key, value in input_specs.items():
        if key not in specs:
            specs[key] = value

    return specs

def populate_species(user_species, specs):
    species_list = ()
    species_dict = {}

    lcdm_species = (
        species.DarkEnergy,
        species.ColdDarkMatter,
        species.Baryon,
        species.Photon,
        species.MasslessNeutrino
    )

    i = 0
    diffrax_vector_idx = 1

    # Add baseline LCDM species if needed.
    if specs["use_LCDM_species"]:
        for s in lcdm_species:
            instance = s(diffrax_vector_idx, specs) # Creates an instance of s. init is now consistent across all species
            species_list = species_list + (instance,)
            species_dict[instance.name] = i

            i += 1
            diffrax_vector_idx += instance.num_equations

    if user_species is not None:
        for s in user_species:
            instance = s(diffrax_vector_idx, specs)
            species_list = species_list + (instance,)
            species_dict[instance.name] = i

            i += 1
            diffrax_vector_idx += instance.num_equations

    return species_list, species_dict

def get_k_axis_perturbations(specs):
    ks = []   # grown dynamically; k_max now scales with delta_l_max

    H0_fid     = specs["H0_fid"]
    tau0_fid   = specs["tau0_fid"]
    rs_rec_fid = specs["rs_rec_fid"]
    k_rec_fid  = 2.*jnp.pi/rs_rec_fid

    k_min = specs["k_min_tau0"] / tau0_fid
    # CLASS raises l_scalar_max by delta_l_max when lensing is on *before*
    # computing k_max, so the transfer integral stays accurate over the whole
    # buffered range. Do the same here.
    l_max_eff = specs["l_max"] + (specs["delta_l_max"] if specs["lensing"] else 0)
    k_max = specs["k_max_tau0_over_l_max"] / tau0_fid * l_max_eff

    k = k_min
    ks.append(k)
    while k < k_max:
        step = (specs["k_step_super"]
                + 0.5 * (jnp.tanh((k-k_rec_fid)/k_rec_fid/specs["k_step_transition"])+1.)
                * (specs["k_step_sub"]-specs["k_step_super"])) * k_rec_fid

        scale2 = H0_fid**2

        step *= (k**2/scale2+1.)/(k**2/scale2+1./specs["k_step_super_reduction"])

        k += step
        ks.append(k)

    specs["k_min"]     = k_min
    specs["k_max_cmb"] = k

    # If the user specified a k_max above the current, we should add these as well.
    if k < specs["k_max"]:
        k_max = specs["k_max"]
        
        while k < k_max:
            step = 0.005

            k += step
            ks.append(k)

    # --- CMB lensing potential extension ------------------------------------
    # CLASS's "full Limber" scheme for C_l^phiphi. The Limber integral samples
    # k = (l+1/2)/chi and therefore runs far past the ceiling the line-of-sight
    # transfer functions need.
    #
    # Only the lensing k-integral sees these modes: the transfer grid still stops
    # at k_max_cmb, and the source spline is restricted to the same range.
    # Modes above this are the lensing-only extension, and get the looser third
    # tolerance tier in perturbations.py. inf when there is no extension, so the
    # user-k_max P(k) modes below never qualify.
    specs["k_limber_start"] = np.inf
    if specs["lensing"]:
        specs["k_limber_start"] = float(k)
        k_max_limber = specs["k_max_limber_over_l_max"] * l_max_eff
        ln_bao_center = np.log(specs["k_bao_center"] * k_rec_fid)
        ln_bao_width  = np.log(specs["k_bao_width"])
        while k < k_max_limber:
            k_per_decade = (
                specs["k_per_decade_for_pk"]
                + (specs["k_per_decade_for_bao"] - specs["k_per_decade_for_pk"])
                * (1. - np.tanh(((np.log(k) - ln_bao_center)/ln_bao_width)**4))
            )
            k = k * 10.**(1./k_per_decade)
            ks.append(k)

    ks = np.asarray(ks)
    # Number of leading nodes that cover the CMB range -- CLASS's k_size_cl. 
    specs["k_size_cmb"] = int(np.searchsorted(ks, specs["k_max_cmb"], side="right"))
    # CLASS takes the transfer q_max straight off the top of that slice
    # (transfer.c: q_max = ppt->k[...k_size_cl-1]). Record it so the transfer
    # grid cannot run past the range the source spline was built on.
    specs["k_max_pert"] = float(ks[specs["k_size_cmb"]-1])
    k_axis_Pk_output = ks[np.where(ks<=specs["k_max"])]

    return jnp.array(ks), jnp.array(k_axis_Pk_output)

def get_k_axis_transfer(specs):
    ks = []   # grown dynamically; no fixed cap

    k_period = 2*jnp.pi/(specs["tau0_fid"] - specs["tau_rec_fid"])

    k = specs["k_min"]
    ks.append(k)
    while k < specs["k_max_cmb"]:
        k = k \
            + k_period * specs["k_transfer_linstep"] * k \
            / (k + specs["k_transfer_linstep"]/specs["k_transfer_logstep"])
        ks.append(k)

    ks = np.asarray(ks)
    # The loop above exits one step *past* k_max_cmb, which would put the final
    # node beyond PT.k[-1] and make the source interpolation in spectrum.py
    # extrapolate (interpax CubicSpline does not clamp). CLASS discards the
    # overshooting node for the same reason (transfer.c: "also checking if we
    # overshot the last point").
    ks = ks[ks <= specs["k_max_pert"]]
    return jnp.array(ks)
