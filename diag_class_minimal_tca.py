"""Who is right on the +0.1% Pi / +0.4% low-l BB: ABCMB (full hierarchy, no TCA)
or CLASS (TCA, forced by its tensor IC)?

Forcing ABCMB tight-coupling resolution (dtmax 0.005, atol 1e-13) did NOT move
BB -> the difference is a converged full-hierarchy-vs-TCA difference in the
moments emerging from the earliest tight-coupling phase, NOT under-resolution.

Decisive: push CLASS's tensor TCA triggers toward 0 (TCA ends ~immediately
after the IC -> CLASS runs the full hierarchy for ~all the evolution). If CLASS
BB rises toward ABCMB as TCA->0, then TCA is the difference and ABCMB (full) is
the accurate one. If CLASS stays put, CLASS is TCA-independent (converged) and
ABCMB is the outlier.

CPU-only, a few CLASS tensor runs. Run under srun.
"""
from classy import Class
import sys, os
import numpy as np
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + '/pytests')
from accuracy_test_bb import PARAMS, R_TENSOR

N_T = -R_TENSOR / 8. * (2. - R_TENSOR / 8. - PARAMS["n_s"])

# ABCMB raw-BB at low-l nodes (from diag_bb_dtmax_test baseline = hp*(1+rel)).
ABCMB_BB = {
    10: 2.43417e-18, 15: 2.22498e-18, 25: 2.04081e-18, 38: 1.75809e-18,
    64: 1.18814e-18, 98: 5.79023e-19,
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
    # high precision k/q/time/tol (hp reference settings)
    "k_step_sub": 0.005, "k_step_super": 0.0002, "q_linstep": 0.05,
    "perturbations_sampling_stepsize": 0.02, "tol_perturbations_integration": 1.e-7,
    "radiation_streaming_trigger_tau_over_tau_k": 1.e4,
}

def tca(trig):
    # smaller trigger -> TCA ends earlier -> closer to full hierarchy
    return {"tight_coupling_trigger_tau_c_over_tau_h": trig,
            "tight_coupling_trigger_tau_c_over_tau_k": trig * 0.67,
            "start_small_k_at_tau_c_over_tau_h": trig * 0.1}

variants = {
    "hp(default TCA)": tca(0.0015),
    "TCA_1e-4": tca(1.e-4),
    "TCA_1e-5": tca(1.e-5),
    "TCA_1e-6": tca(1.e-6),
    "TCA_1e-7": tca(1.e-7),
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
        print(f"{name}: FAILED - {repr(e)[:120]}", flush=True)
    finally:
        M.struct_cleanup(); M.empty()

print("\n--- CLASS BB at low-l nodes as TCA->0  (does it rise toward ABCMB?) ---")
print("  l   | " + "".join(f"{n[:13]:>14s} " for n in results) + "|   ABCMB")
for L in NODES:
    row = f" {L:4d} | "
    for n in results:
        row += f"{results[n][L]:.5e} "
    row += f"|  {ABCMB_BB[L]:.5e}"
    print(row)

print("\n--- ABCMB / CLASS_variant  (->1 means CLASS agrees with ABCMB) ---")
print("  l   | " + "".join(f"{n[:13]:>14s} " for n in results))
for L in NODES:
    row = f" {L:4d} | "
    for n in results:
        row += f"{ABCMB_BB[L]/results[n][L]:.5f}      "
    print(row)
print("DONE", flush=True)
