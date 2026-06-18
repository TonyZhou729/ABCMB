"""CLASS 3.3.4 convergence ladder, part 2: bessel interp + extreme k/q."""
from classy import Class
import sys, os
file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir + '/pytests')
from accuracy_test_bb import PARAMS, R_TENSOR

ABCMB_BB = {237: 1.984247e-20, 296: 1.069961e-20, 450: 2.144051e-21,
            490: 1.246576e-21}

base = {
    "output": "tCl, pCl",
    "modes": "t",
    "r": R_TENSOR,
    "n_t": -0.012782500000000002,
    "l_max_tensors": 500,
    "lensing": "no",
    "H0": PARAMS["h"] * 100,
    "omega_b": PARAMS["omega_b"],
    "omega_cdm": PARAMS["omega_cdm"],
    "A_s": PARAMS["A_s"],
    "N_ur": PARAMS["Neff"],
    "YHe": PARAMS["YHe"],
    "N_ncdm": 0,
    "reio_parametrization": "reio_camb",
    "tau_reio": PARAMS["tau_reion"],
    "reionization_width": PARAMS["Delta_z_reion"],
    "helium_fullreio_redshift": PARAMS["z_reion_He"],
    "helium_fullreio_width": PARAMS["Delta_z_reion_He"],
    "reionization_exponent": PARAMS["exp_reion"],
}

KQ3 = {"k_step_sub": 0.005, "k_step_super": 0.0002, "q_linstep": 0.05}
TIME = {"perturbations_sampling_stepsize": 0.02,
        "tol_perturbations_integration": 1.e-7}
APPROX = {"tight_coupling_trigger_tau_c_over_tau_h": 0.0015,
          "tight_coupling_trigger_tau_c_over_tau_k": 0.001,
          "start_small_k_at_tau_c_over_tau_h": 0.00015,
          "radiation_streaming_trigger_tau_over_tau_k": 1.e4}
L4 = {**KQ3, **TIME, **APPROX}

variants = {
    "L4b_hyper": {**L4, "hyper_sampling_flat": 32.},
    "L5_all": {**L4,
               "k_step_sub": 0.002, "k_step_super": 0.0001,
               "q_linstep": 0.02,
               "hyper_sampling_flat": 32.,
               "start_sources_at_tau_c_over_tau_h": 0.004,
               "transfer_neglect_delta_k_T_t2": 100.,
               "transfer_neglect_delta_k_T_e": 100.,
               "transfer_neglect_delta_k_T_b": 100.},
}

for name, extra in variants.items():
    M = Class()
    M.set({**base, **extra})
    try:
        M.compute()
    except Exception as e:
        print(f"{name}: FAILED — {e}")
        continue
    bb = M.raw_cl(500)["bb"]
    line = f"{name:12s}"
    for l in [237, 296, 450, 490]:
        line += f"  l{l}:{ABCMB_BB[l]/bb[l]:.4f}"
    print(line)
    M.struct_cleanup()
    M.empty()
print("DONE")
