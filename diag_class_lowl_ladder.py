"""Is CLASS-hp actually CONVERGED at LOW ell, or is the BB 'excess' really
CLASS under-convergence there?

The prior diag_class_ladder.py only checked HIGH-ell nodes (237..490). The
low-ell BB excess (peak +0.4% at l~10) has never been tested for CLASS-side
convergence. ABCMB integrates the full stiff tensor hierarchy with NO TCA at
tight tol; CLASS-hp still uses early TCA + finite q/k sampling. If CLASS bb at
low ell DRIFTS UP toward ABCMB as precision is pushed past the current 'hp'
settings, then ABCMB is closer to truth and the 'excess' is a CLASS artifact.
If CLASS is stable across rungs, the excess is real on the ABCMB side.

CPU-only (CLASS), a handful of tensor runs (~30-60s each). Run under srun.
"""
from classy import Class
import sys, os
import numpy as np
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + '/pytests')
from accuracy_test_bb import PARAMS, R_TENSOR

N_T = -R_TENSOR / 8. * (2. - R_TENSOR / 8. - PARAMS["n_s"])
print(f"n_t = {N_T}", flush=True)

# ABCMB raw-BB at low-ell nodes (reconstructed from diag logs: hp*(1+relerr_x1)).
# Fixed reference; we watch whether ABCMB/CLASS_variant -> 1 as CLASS precision
# increases.
ABCMB_BB = {
    10: 2.43417e-18, 15: 2.22498e-18, 21: 2.12433e-18, 25: 2.04081e-18,
    31: 1.91107e-18, 38: 1.75809e-18, 47: 1.55798e-18, 58: 1.31566e-18,
    64: 1.18814e-18, 71: 1.04453e-18, 79: 8.91103e-19, 88: 7.33357e-19,
    98: 5.79023e-19, 109: 4.35091e-19, 122: 2.99790e-19,
}
NODES = sorted(ABCMB_BB)

base = {
    "output": "tCl, pCl", "modes": "t", "r": R_TENSOR, "n_t": N_T,
    "l_max_tensors": 500, "lensing": "no",
    "H0": PARAMS["h"] * 100, "omega_b": PARAMS["omega_b"],
    "omega_cdm": PARAMS["omega_cdm"], "A_s": PARAMS["A_s"],
    "N_ur": PARAMS["Neff"], "YHe": PARAMS["YHe"], "N_ncdm": 0,
    "reio_parametrization": "reio_camb", "tau_reio": PARAMS["tau_reion"],
    "reionization_width": PARAMS["Delta_z_reion"],
    "helium_fullreio_redshift": PARAMS["z_reion_He"],
    "helium_fullreio_width": PARAMS["Delta_z_reion_He"],
    "reionization_exponent": PARAMS["exp_reion"],
}

# current hp settings (= class_tensor_hp_reference / ladder L4)
HP = {
    "k_step_sub": 0.005, "k_step_super": 0.0002, "q_linstep": 0.05,
    "perturbations_sampling_stepsize": 0.02,
    "tol_perturbations_integration": 1.e-7,
    "tight_coupling_trigger_tau_c_over_tau_h": 0.0015,
    "tight_coupling_trigger_tau_c_over_tau_k": 0.001,
    "start_small_k_at_tau_c_over_tau_h": 0.00015,
    "radiation_streaming_trigger_tau_over_tau_k": 1.e4,
}
# push EVERYTHING further: denser k/q, finer time, tighter tol, weaker TCA
ULTRA = {
    "k_step_sub": 0.002, "k_step_super": 0.00005, "q_linstep": 0.015,
    "perturbations_sampling_stepsize": 0.008,
    "tol_perturbations_integration": 1.e-8,
    "tight_coupling_trigger_tau_c_over_tau_h": 0.0004,
    "tight_coupling_trigger_tau_c_over_tau_k": 0.0003,
    "start_small_k_at_tau_c_over_tau_h": 0.00004,
    "radiation_streaming_trigger_tau_over_tau_k": 1.e5,
}
# isolate which axis matters, vs HP
HP_qonly = {**HP, "q_linstep": 0.015}          # denser neutrino momentum only
HP_tcaonly = {**HP,                             # weaker TCA only
              "tight_coupling_trigger_tau_c_over_tau_h": 0.0004,
              "tight_coupling_trigger_tau_c_over_tau_k": 0.0003,
              "start_small_k_at_tau_c_over_tau_h": 0.00004}
HP_konly = {**HP, "k_step_sub": 0.002, "k_step_super": 0.00005}  # denser k only

variants = {
    "L0_default": {},
    "L4_hp": HP,
    "HP_qonly(q0.015)": HP_qonly,
    "HP_tcaonly": HP_tcaonly,
    "HP_konly": HP_konly,
    "L5_ultra": ULTRA,
}

results = {}
for name, extra in variants.items():
    M = Class()
    M.set({**base, **extra})
    try:
        M.compute()
        results[name] = np.asarray(M.raw_cl(500)["bb"])
        print(f"{name}: done", flush=True)
    except Exception as e:
        print(f"{name}: FAILED - {e}", flush=True)
    finally:
        M.struct_cleanup(); M.empty()

print("\n--- ABCMB / CLASS_variant  at low-ell nodes (ratio; ->1 means CLASS "
      "agrees w/ ABCMB) ---")
hdr = "  l   " + "".join(f"| {n[:11]:>11s} " for n in results)
print(hdr)
for L in NODES:
    row = f" {L:4d} "
    for n in results:
        row += f"|   {ABCMB_BB[L]/results[n][L]:.4f}  "
    print(row)

print("\n--- CLASS self-convergence: bb[L] across rungs (watch if it drifts "
      "up toward ABCMB) ---")
print("  l   | " + "".join(f"{n[:11]:>13s} " for n in results) + "   ABCMB")
for L in NODES:
    row = f" {L:4d} | "
    for n in results:
        row += f"{results[n][L]:.5e} "
    row += f"  {ABCMB_BB[L]:.5e}"
    print(row)
print("DONE", flush=True)
