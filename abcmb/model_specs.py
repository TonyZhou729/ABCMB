import numpy as np
import jax.numpy as jnp
import equinox as eqx

from . import species
from . import constants as cnst

def load_specs(input_specs):

    specs = {}

    specs["use_LCDM_species"] = input_specs.get("use_LCDM_species", True)

    ### CURVATURE ###
    # curvature: use the hyperspherical-Bessel recurrence (required for omega_k != 0).
    # omega_k_ref: static reference Omega_k h^2 for the (shape-static) k-grids;
    # closed runs require it (grid starts at nu=3, k=sqrt(8K)), set to the
    # most-closed value explored. closed_integer_nu: snap the low-q transfer
    # grid to the integer-nu lattice (see get_k_axis_transfer). params['omega_k']
    # carries the actual traced value.
    specs["curvature"]   = input_specs.get("curvature", False)
    specs["omega_k_ref"] = input_specs.get("omega_k_ref", 0.)
    specs["closed_integer_nu"] = input_specs.get("closed_integer_nu", True)

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
        species.MasslessNeutrino,
        species.Curvature
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
    ks = np.zeros(2000)

    H0_fid     = specs["H0_fid"]
    tau0_fid   = specs["tau0_fid"]
    rs_rec_fid = specs["rs_rec_fid"]
    k_rec_fid  = 2.*jnp.pi/rs_rec_fid

    k_min = specs["k_min_tau0"] / tau0_fid
    k_max = specs["k_max_tau0_over_l_max"] / tau0_fid * specs["l_max"]

    # Static curvature reference for the grid (CLASS perturb_get_k_list): closed
    # starts at nu=3 (k=sqrt(8K)) and needs k_max/angular_rescaling to reach the
    # same l_max; open starts at k just above sqrt(|K|) (q -> 0).
    K_ref = -specs["omega_k_ref"] * (cnst.H0_over_h/cnst.c_Mpc_over_s)**2
    specs["K_ref"] = K_ref
    if K_ref > 0.:
        # closed: divide k_max by the angular rescaling sin(s)/s (s = sqrt(K_ref)
        # chi_*) so l_max is still reached after the S_K(chi) distance shrink.
        s = np.sqrt(K_ref) * (tau0_fid - specs["tau_rec_fid"])
        k_min = np.sqrt((8.-1.e-4)*K_ref)
        k_max = k_max / (np.sin(s)/s)
    elif K_ref < 0.:
        k_min = np.sqrt(-K_ref + k_min**2)

    k = k_min
    ks[0] = k
    i = 0
    while k < k_max:
        step = (specs["k_step_super"]
                + 0.5 * (jnp.tanh((k-k_rec_fid)/k_rec_fid/specs["k_step_transition"])+1.)
                * (specs["k_step_sub"]-specs["k_step_super"])) * k_rec_fid

        # CLASS adds |K| to the super-Hubble densification scale in curved space.
        scale2 = H0_fid**2 + abs(K_ref)

        step *= (k**2/scale2+1.)/(k**2/scale2+1./specs["k_step_super_reduction"])

        k += step
        i += 1
        ks[i] = k

    specs["k_min"]     = k_min
    specs["k_max_cmb"] = k

    # If lensing is needed, we need to extend max k by some amount to accurately compute high-l lensing.
    if specs["lensing"]:
        k_max = k + 0.3
        
        while k < k_max:
            step = 0.005

            k += step
            i += 1
            ks[i] = k

    # If the user specified a k_max above the current, we should add these as well.
    if k < specs["k_max"]:
        k_max = specs["k_max"]
        
        while k < k_max:
            step = 0.005

            k += step
            i += 1
            ks[i] = k

    ks = ks[np.where(ks>0)]
    k_axis_Pk_output = ks[np.where(ks<=specs["k_max"])]

    return jnp.array(ks), jnp.array(k_axis_Pk_output)

def get_k_axis_transfer(specs):
    ks = np.zeros(8000)

    k_period = 2*jnp.pi/(specs["tau0_fid"] - specs["tau_rec_fid"])

    K_ref = specs.get("K_ref", 0.)

    if K_ref < 0.:
        # Open universes: transfer functions oscillate in q (q^2 = k^2 + K),
        # so walk the step formula in q and map k = sqrt(q^2 - K) (CLASS
        # densifies its low-q sampling the same way).
        q = specs["k_min_tau0"] / specs["tau0_fid"]
        q_max = np.sqrt(specs["k_max_cmb"]**2 + K_ref)
        qs = [q]
        while q < q_max:
            q = q \
                + k_period * specs["k_transfer_linstep"] * q \
                / (q + specs["k_transfer_linstep"]/specs["k_transfer_logstep"])
            qs.append(q)
        ks = np.sqrt(np.array(qs)**2 - K_ref)
        return jnp.array(ks)

    k = specs["k_min"]
    ks[0] = k
    i = 0
    while k < specs["k_max_cmb"]:
        k = k \
            + k_period * specs["k_transfer_linstep"] * k \
            / (k + specs["k_transfer_linstep"]/specs["k_transfer_logstep"])
        i += 1
        ks[i] = k

    ks = ks[np.where(ks>0)]

    # Closed universes have discrete modes nu = sqrt(k^2+K)/sqrt(K) = 3,4,5,...
    # Resample the grid's low end onto the integer-nu lattice (CLASS
    # transfer_get_q_list): Delta nu = 1 up to nu_dense_top, then ramp.
    K_ref = specs.get("K_ref", 0.)
    if K_ref > 0. and specs.get("closed_integer_nu", True):
        # Delta nu = 1 up to nu_dense_top (samples the sharp ell ~ nu onset and
        # makes the k-trapezoid the exact discrete-mode sum), then ramp the step
        # over n_transition points to avoid a sampling-density boundary artifact.
        sqrtK = np.sqrt(K_ref)
        nu_dense_top = 512.
        n_transition = 250.
        nu = 3.
        nus = [nu]
        i_since = 0
        while True:
            k_cur = np.sqrt(max(nu**2*K_ref - K_ref, 0.))
            if k_cur >= specs["k_max_cmb"]:
                break
            step_k = k_period * specs["k_transfer_linstep"] * max(k_cur, sqrtK) \
                / (max(k_cur, sqrtK) + specs["k_transfer_linstep"]/specs["k_transfer_logstep"])
            dnu_f = max(1., np.round(step_k/sqrtK))
            if nu < nu_dense_top:
                dnu = 1.
            else:
                t = min(1., i_since/n_transition)
                i_since += 1
                dnu = max(1., np.round((1.-t)*1. + t*dnu_f))
            nu += dnu
            nus.append(nu)
        nu_grid = np.array(nus)
        ks = np.sqrt(nu_grid**2*K_ref - K_ref)

    return jnp.array(ks)