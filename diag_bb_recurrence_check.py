"""
Resolve whether the every-ell raw-BB residual (~5e-3 at l=387, 7e-3-class at
l=477) after the Bessel-recurrence port is ABCMB error or CLASS's own
interpolation between its sparse transfer ell-nodes.

Compares ABCMB recurrence ClBB (every ell, exact) against the high-precision
tensor-only CLASS reference, split by:
  - CLASS computed ell-nodes (class_lnodes_tmp.txt) -> both codes exact here,
    so this is the true ABCMB transfer-accuracy test.
  - every ell -> includes CLASS-interpolated mid-node ells (l=387, 477 sit in
    CLASS's 40-wide node gaps 370-410 and 450-490).
If node-level is sub-permille while every-ell peaks at the gap midpoints, the
5e-3 is CLASS interpolation, not ABCMB.  No jax_debug_nans (forward only).
"""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from abcmb.main import Model
from classy import Class

ELLMIN, ELLMAX, R = 2, 2500, 0.1
PARAMS = {
    'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225, 'A_s': 2.12424e-9,
    'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245, 'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5, 'z_reion_He': 3.5,
    'Delta_z_reion_He': 0.5, 'exp_reion': 1.5, 'r': R,
}


def class_tensor_hp(l_tensor_max=500):
    n_t_scc = -R/8.*(2. - R/8. - PARAMS["n_s"])
    M = Class()
    M.set({
        "output": "tCl, pCl", "modes": "t", "r": R, "n_t": n_t_scc,
        "l_max_tensors": l_tensor_max, "lensing": "no",
        "H0": PARAMS["h"]*100, "omega_b": PARAMS["omega_b"],
        "omega_cdm": PARAMS["omega_cdm"], "A_s": PARAMS["A_s"],
        "N_ur": PARAMS["Neff"], "YHe": PARAMS["YHe"], "N_ncdm": 0,
        "reio_parametrization": "reio_camb", "tau_reio": PARAMS["tau_reion"],
        "reionization_width": PARAMS["Delta_z_reion"],
        "helium_fullreio_redshift": PARAMS["z_reion_He"],
        "helium_fullreio_width": PARAMS["Delta_z_reion_He"],
        "reionization_exponent": PARAMS["exp_reion"],
        "k_step_sub": 0.005, "k_step_super": 0.0002, "q_linstep": 0.05,
        "perturbations_sampling_stepsize": 0.02,
        "tol_perturbations_integration": 1.e-7,
        "tight_coupling_trigger_tau_c_over_tau_h": 0.0015,
        "tight_coupling_trigger_tau_c_over_tau_k": 0.001,
        "start_small_k_at_tau_c_over_tau_h": 0.00015,
        "radiation_streaming_trigger_tau_over_tau_k": 1.e4,
    })
    M.compute()
    return M.raw_cl(l_tensor_max)


model = Model(l_max=ELLMAX, lensing=False, tensors=True, l_max_g=12, l_max_pol_g=10)
out = model(PARAMS)
clbb = np.asarray(out.ClBB)
ells = np.arange(ELLMIN, ELLMAX+1)

cl_hp = class_tensor_hp()
bb_hp = np.zeros(ELLMAX - ELLMIN + 1)
bb_hp[:500 - ELLMIN + 1] = cl_hp["bb"][ELLMIN:]

# CLASS's computed transfer multipoles up to ~500 (the CLASS l-sampling,
# formerly abcmb/bessel_tab/l.txt). CLASS interpolates raw_cl between these.
nodes = np.array([
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 23, 25,
    28, 31, 34, 38, 42, 47, 52, 58, 64, 71, 79, 88, 98, 109, 122, 136, 152,
    170, 190, 212, 237, 265, 296, 331, 370, 410, 450, 490,
])
denom = np.where(bb_hp != 0., bb_hp, 1.)
err = np.abs(clbb - bb_hp) / np.abs(denom)


def report(lo, hi, label):
    m_all = (ells >= lo) & (ells <= hi)
    m_node = m_all & np.isin(ells, nodes)
    amax_all = ells[m_all][err[m_all].argmax()]
    amax_nd = ells[m_node][err[m_node].argmax()]
    print(f"{label:14s} every-l: {err[m_all].max():.3e} (l={amax_all})   "
          f"CLASS-nodes only: {err[m_node].max():.3e} (l={amax_nd})")


print("\n==== raw tensor BB: ABCMB recurrence vs CLASS hp ====")
report(100, 490, "recomb")
report(3, 100, "low-l")
for L in (387, 477):
    print(f"  l={L}: rel err {err[ells == L][0]:.3e}  "
          f"(in CLASS gap {'370-410' if L < 410 else '450-490'})")
print(f"  sliver 491-500 every-l max: "
      f"{err[(ells > 490) & (ells <= 500)].max():.3e}")
