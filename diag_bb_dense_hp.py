"""ABCMB raw BB (GPU) vs CLASS hp + dense-ell (every l computed exactly).

Isolates ABCMB's ell-spline error: at nodes the transfer agreement is
~1e-3; mid-interval residuals are ABCMB spline error alone (CLASS side
has no interpolation here).
"""
import sys, os
file_dir = '/pscratch/sd/c/carag/ABCMB-bmodes'
sys.path.insert(0, file_dir)
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from abcmb.main import Model
sys.path.insert(0, file_dir + '/pytests')
from accuracy_test_bb import PARAMS, R_TENSOR, ELLMIN, ELLMAX

model = Model(l_max=ELLMAX, lensing=False, tensors=True, l_max_g=12, l_max_pol_g=10)
output = model(PARAMS)
ours = np.asarray(output.ClBB)
print("ABCMB done", flush=True)

from classy import Class
n_t_scc = -R_TENSOR / 8. * (2. - R_TENSOR / 8. - PARAMS["n_s"])
M = Class()
M.set({
    "output": "tCl, pCl",
    "modes": "t",
    "r": R_TENSOR,
    "n_t": n_t_scc,
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
    "k_step_sub": 0.005,
    "k_step_super": 0.0002,
    "q_linstep": 0.05,
    "perturbations_sampling_stepsize": 0.02,
    "tol_perturbations_integration": 1.e-7,
    "tight_coupling_trigger_tau_c_over_tau_h": 0.0015,
    "tight_coupling_trigger_tau_c_over_tau_k": 0.001,
    "start_small_k_at_tau_c_over_tau_h": 0.00015,
    "radiation_streaming_trigger_tau_over_tau_k": 1.e4,
    "l_linstep": 1,
    "l_logstep": 1.0,
})
M.compute()
bb = M.raw_cl(500)["bb"]
print("CLASS hp dense done", flush=True)

ells = np.arange(ELLMIN, 501)
theirs = bb[ELLMIN:]
err = np.abs(ours[:len(ells)] - theirs) / np.abs(theirs)

np.savez(file_dir + "/diag_bb_dense_hp.npz", ells=ells,
         abcmb=ours[:len(ells)], class_hp_dense=theirs)

for lo, hi in [(3, 100), (100, 200), (200, 300), (300, 400), (400, 450),
               (450, 491), (491, 501)]:
    s = (ells >= lo) & (ells < hi)
    print(f"max err in [{lo},{hi}): {err[s].max():.4f} at l={ells[s][err[s].argmax()]}")
print(f"l=2 err: {err[0]:.4f}")
print("DONE")
