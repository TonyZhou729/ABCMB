"""Final discriminator: is the tensor-BB polarization-source bias a
RECOMBINATION-code (HyRex vs CLASS) difference shared with scalar polarization,
or tensor-specific?

The visibility g is shared scalar<->tensor. Tensor finding: source sqrt6*g*Pi
is biased (~0.1% Pi high near recomb; g wings differ ~0.5%), while h is exact.
If recomb-driven, ABCMB SCALAR EE (also = g * scalar-quadrupole) should show
the same recomb fingerprint vs high-precision CLASS in the recomb-sourced band
(l~100-1500), and TT (less g-sensitive) should be cleaner -- mirroring the
tensor EE/BB-high, TT-less pattern. If scalar EE is clean, the bias is
tensor-specific.

ABCMB scalar (tensors=False, faster) vs high-precision SCALAR CLASS. Report
TT/TE/EE rel err vs l, and mean EE offset over the recomb band.

GPU, ~3 min warm (no tensor PE).
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
import sys
file_dir = "/pscratch/sd/c/carag/ABCMB-bmodes"
sys.path.insert(0, file_dir)
sys.path.insert(0, file_dir + "/pytests")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from classy import Class
from abcmb.main import Model
from accuracy_test_bb import PARAMS, ELLMIN, ELLMAX

model = Model(l_max=ELLMAX, lensing=False, tensors=False,
              l_max_g=12, l_max_pol_g=10)
out = model(PARAMS)
tt_ab = np.asarray(out.ClTT); te_ab = np.asarray(out.ClTE); ee_ab = np.asarray(out.ClEE)
print("ABCMB scalar done", flush=True)

M = Class()
M.set({
    "output": "tCl,pCl", "modes": "s", "l_max_scalars": ELLMAX, "lensing": "no",
    "H0": PARAMS["h"] * 100, "omega_b": PARAMS["omega_b"],
    "omega_cdm": PARAMS["omega_cdm"], "A_s": PARAMS["A_s"], "n_s": PARAMS["n_s"],
    "N_ur": PARAMS["Neff"], "YHe": PARAMS["YHe"], "N_ncdm": 0,
    "reio_parametrization": "reio_camb", "tau_reio": PARAMS["tau_reion"],
    "reionization_width": PARAMS["Delta_z_reion"],
    "helium_fullreio_redshift": PARAMS["z_reion_He"],
    "helium_fullreio_width": PARAMS["Delta_z_reion_He"],
    "reionization_exponent": PARAMS["exp_reion"],
    "l_max_g": model.specs["l_max_g"], "l_max_pol_g": model.specs["l_max_pol_g"],
    "l_max_ur": model.specs["l_max_massless_nu"],
    # high precision scalars
    "k_step_sub": 0.015, "k_step_super": 0.0008, "k_per_decade_for_pk": 30,
    "perturbations_sampling_stepsize": 0.02,
    "tol_perturbations_integration": 1.e-7,
    "radiation_streaming_trigger_tau_over_tau_k": 1.e4,
})
M.compute()
cl = M.raw_cl(ELLMAX)
tt_cl = cl["tt"][ELLMIN:]; te_cl = cl["te"][ELLMIN:]; ee_cl = cl["ee"][ELLMIN:]
M.struct_cleanup(); M.empty()
print("CLASS scalar hp done", flush=True)

ells = np.arange(ELLMIN, ELLMAX + 1)
def rel(a, c):
    return np.where(c != 0, (a - c) / np.where(c != 0, c, 1.), np.nan)
r_tt = rel(tt_ab, tt_cl); r_te = rel(te_ab, te_cl); r_ee = rel(ee_ab, ee_cl)

print("\n--- scalar TT / EE rel err vs CLASS-hp (ABCMB/CLASS - 1) ---")
print("   l   |   TT err    |   EE err")
for L in [30, 50, 80, 100, 150, 200, 300, 400, 500, 700, 900, 1100, 1300, 1500]:
    i = L - ELLMIN
    print(f" {L:5d} | {r_tt[i]:+.3e} | {r_ee[i]:+.3e}")

# recomb-sourced EE band (avoid reion bump at l<20): mean offset
band = (ells >= 100) & (ells <= 1500)
print(f"\n[summary] recomb band 100<=l<=1500:")
print(f"  mean EE rel err = {np.nanmean(r_ee[band]):+.3e}  "
      f"(max |EE| = {np.nanmax(np.abs(r_ee[band])):.3e})")
print(f"  mean TT rel err = {np.nanmean(r_tt[band]):+.3e}  "
      f"(max |TT| = {np.nanmax(np.abs(r_tt[band])):.3e})")
# low-l EE (reion-ish) for completeness
band2 = (ells >= 30) & (ells < 100)
print(f"  mean EE rel err 30<=l<100 = {np.nanmean(r_ee[band2]):+.3e}")
print("DONE", flush=True)
