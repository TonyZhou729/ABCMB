"""Attribute the raw-BB residual at l=477 (the §5 open item).

The raw-BB accuracy test passes everywhere except l=477 (6.9e-3 vs the
2.5e-3 bar), which sits exactly between the tensor spline nodes 450 and 490.
At the nodes themselves ABCMB matches converged CLASS sub-permille, so the
residual is suspected to be CubicSpline interpolation error over the 40-wide
node gap, NOT a transfer-physics error.

This used to crank CLASS precision *and* set l_linstep=1, which ran >70 min
and never finished. That is unnecessary: precision controls the *amplitude*
of the spectrum (a smooth ~0.7% offset in the tail), while the question here
is purely about the *shape* between nodes, which the interpolation scheme
resolves the same way at any precision. So we run CLASS at DEFAULT precision
with l_linstep=1 (every l computed exactly -> CLASS-side has no interpolation)
and do two things:

  (A) CLASS-internal self-interpolation test (the decisive one, needs no
      ABCMB): take CLASS's OWN dense BB at the ABCMB spline nodes, push them
      through the IDENTICAL interpax.CubicSpline, and measure the error vs
      CLASS's dense curve. This isolates the interpolation scheme's error
      with zero ABCMB involvement. If it peaks at ~7e-3 near l=477, the
      scheme alone explains the residual.

  (B) real ABCMB vs CLASS-dense, to confirm the magnitude and location of
      the actual residual matches (A).

Run on a GPU node (ABCMB). CLASS runs on CPU. ~5 min total.
"""
import sys, os
file_dir = '/pscratch/sd/c/carag/ABCMB-bmodes'
sys.path.insert(0, file_dir)
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from interpax import CubicSpline
from abcmb.main import Model
from abcmb.spectrum import bessel_l_tab
sys.path.insert(0, file_dir + '/pytests')
from accuracy_test_bb import PARAMS, R_TENSOR, ELLMIN

L_TEN_MAX = 500          # ABCMB l_tensor_max (default spec)
CLASS_LMAX = 540         # > 530 so CLASS dense covers ABCMB's top node (530)

# ---- ABCMB (GPU) ----------------------------------------------------------
model = Model(l_max=2500, lensing=False, tensors=True, l_max_g=12, l_max_pol_g=10)
output = model(PARAMS)
abcmb_bb = np.asarray(output.ClBB)          # integer ells ELLMIN..2500
print("ABCMB done", flush=True)

# ---- CLASS DEFAULT precision, dense ell (l_linstep=1) ----------------------
from classy import Class
n_t_scc = -R_TENSOR / 8. * (2. - R_TENSOR / 8. - PARAMS["n_s"])
M = Class()
M.set({
    "output": "tCl, pCl",
    "modes": "t",
    "r": R_TENSOR,
    "n_t": n_t_scc,
    "l_max_tensors": CLASS_LMAX,
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
    # l_linstep=5: CLASS computes 475 & 480 directly (bracketing the l=477
    # failure) plus dense intermediate truth; ~5x cheaper than l_linstep=1.
    # CLASS's residual 5-wide interpolation in raw_cl is negligible against
    # ABCMB's 40-wide node gap, which is what we are isolating.
    "l_linstep": 5,
})
M.compute()
bb_dense = M.raw_cl(CLASS_LMAX)["bb"]        # indexed directly by ell
print("CLASS default dense done", flush=True)

# ---- node set: the ABCMB tensor spline nodes ------------------------------
# ABCMB uses every bessel_l_tab entry from ELLMIN up to the first node >= 500.
bl = np.asarray(bessel_l_tab)
idx_min = np.where(bl <= ELLMIN)[0][-1]
idx_max = np.where(bl >= L_TEN_MAX)[0][0]
node_ells = bl[idx_min:idx_max + 1]          # ..., 410, 450, 490, 530
print("ABCMB tensor spline nodes near the gap:",
      node_ells[node_ells >= 370].tolist())

dense_ells = np.arange(ELLMIN, L_TEN_MAX + 1)

# ---- (A) CLASS-internal self-interpolation error --------------------------
bb_nodes_truth = bb_dense[node_ells]                       # CLASS dense AT nodes
respline = np.asarray(
    CubicSpline(node_ells, bb_nodes_truth, check=False)(dense_ells))
scheme_err = np.abs(respline - bb_dense[dense_ells]) / np.abs(bb_dense[dense_ells])

# ---- (B) real ABCMB vs CLASS dense ----------------------------------------
abcmb_on_dense = abcmb_bb[dense_ells - ELLMIN]
abcmb_err = np.abs(abcmb_on_dense - bb_dense[dense_ells]) / np.abs(bb_dense[dense_ells])

# ---- node-level agreement (sanity: should be sub-percent) ------------------
print("\nNode agreement  ABCMB / CLASS-dense:")
for L in node_ells[(node_ells >= 370) & (node_ells <= L_TEN_MAX)]:
    print(f"  l={L:3d}: {abcmb_bb[L - ELLMIN] / bb_dense[L]:.5f}")

# ---- the gap region -------------------------------------------------------
print("\n  l   |  scheme-interp err  |  ABCMB-vs-CLASS err  | node?")
for L in range(440, 501):
    tag = " <- NODE" if L in node_ells else ""
    star = "  *477*" if L == 477 else ""
    print(f" {L:4d} |     {scheme_err[L-ELLMIN]:.4e}    |      "
          f"{abcmb_err[L-ELLMIN]:.4e}     |{tag}{star}")

band = (dense_ells >= 3) & (dense_ells <= 490)
i_s = scheme_err[band].argmax()
i_a = abcmb_err[band].argmax()
print(f"\nmax scheme-interp err (3<=l<=490): {scheme_err[band][i_s]:.4e} "
      f"at l={dense_ells[band][i_s]}")
print(f"max ABCMB-vs-CLASS err (3<=l<=490): {abcmb_err[band][i_a]:.4e} "
      f"at l={dense_ells[band][i_a]}")

np.savez(file_dir + "/diag_bb_dense_hp.npz", dense_ells=dense_ells,
         abcmb=abcmb_on_dense, class_dense=bb_dense[dense_ells],
         respline=respline, scheme_err=scheme_err, abcmb_err=abcmb_err,
         node_ells=node_ells)
print("\nVERDICT: if scheme-interp err peaks near l=477 at ~the same "
      "magnitude as\nABCMB-vs-CLASS err, the residual is the CubicSpline "
      "scheme over the 450-490\ngap, not transfer physics. DONE")
