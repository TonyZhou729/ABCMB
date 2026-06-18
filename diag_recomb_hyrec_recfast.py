"""Both ABCMB (HyRex) and default CLASS (HYREC-2) use HYREC-2-family recomb, yet
ABCMB's visibility differs ~0.5% in the wings -> a genuine HyRex vs C-HYREC-2
discrepancy. Pin it:

  - Is ABCMB's xe(z) off (recomb SOLVE), or does xe match but g differ
    (VISIBILITY / optical-depth computation)?
  - Does ABCMB BB match CLASS-HyRec or CLASS-RecFast better at low l?

Compare ABCMB xe(z), g(z), BB-nodes against CLASS tensor-hp run with
recombination=HyRec (explicit) AND recombination=recfast.

GPU: one ABCMB tensors=True build + 2 CLASS tensor-hp runs. ~6 min warm.
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
import sys
file_dir = "/pscratch/sd/c/carag/ABCMB-bmodes"
sys.path.insert(0, file_dir); sys.path.insert(0, file_dir + "/pytests")
import jax
from jax import vmap
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from classy import Class
from abcmb.main import Model
from abcmb.spectrum import bessel_l_tab
from accuracy_test_bb import PARAMS, ELLMIN, ELLMAX

# ---- ABCMB ----
model = Model(l_max=ELLMAX, lensing=False, tensors=True, l_max_g=12, l_max_pol_g=10)
out = model(PARAMS)
BG, params = out.BG, out.params
TPT = jax.block_until_ready(model.TPE.full_evolution((BG, params)))
TSS = model.TSS
bb_ab = np.asarray(jax.block_until_ready(vmap(TSS.Cl_one_ell, in_axes=(0, None, None, None))(
    TSS.tensor_ells_indices, TPT, BG, params)[3]))
node_ells = np.asarray(bessel_l_tab)[np.asarray(TSS.tensor_ells_indices)]
print("ABCMB done", flush=True)

z = np.linspace(800, 1500, 700)
lna = jnp.asarray(-np.log(1.0 + z))
xe_ab = np.asarray(vmap(BG.xe)(lna))
g_ab = np.asarray(vmap(BG.visibility, in_axes=[0, None])(lna, params))

def run_class(recomb):
    M = Class()
    M.set({
        "output": "tCl,pCl", "modes": "t", "r": PARAMS["r"], "n_t": -0.0127075,
        "l_max_tensors": 500, "lensing": "no", "recombination": recomb,
        "H0": PARAMS["h"] * 100, "omega_b": PARAMS["omega_b"],
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
    bb = np.asarray(M.raw_cl(500)["bb"])
    th = M.get_thermodynamics()
    zc = np.asarray(th["z"]); o = np.argsort(zc)
    xe = np.interp(z, zc[o], np.asarray(th["x_e"])[o])
    gk = next(k for k in th if k.strip().lower().startswith("g "))
    g = np.interp(z, zc[o], np.asarray(th[gk])[o])
    M.struct_cleanup(); M.empty()
    return bb, xe, g

bb_hy, xe_hy, g_hy = run_class("HyRec");  print("CLASS HyRec done", flush=True)
bb_rf, xe_rf, g_rf = run_class("recfast"); print("CLASS RecFast done", flush=True)

print("\n--- ABCMB BB / CLASS BB at low-l nodes (HyRec vs RecFast reference) ---")
print("  l   |  /HyRec   | /RecFast")
for L in node_ells[node_ells <= 130]:
    i = np.where(node_ells == L)[0][0]
    print(f" {L:4d} |  {bb_ab[i]/bb_hy[L]:.4f}  |  {bb_ab[i]/bb_rf[L]:.4f}")

print("\n--- xe(z): ABCMB vs CLASS-HyRec vs CLASS-RecFast (rel to HyRec) ---")
print("   z   |  xe_ABCMB    xe/HyRec-1   xe_RecFast/HyRec-1 |  g_ABCMB/HyRec-1  g_RF/HyRec-1")
for zi in [850, 950, 1000, 1050, 1080, 1100, 1150, 1200, 1300, 1450]:
    j = int(np.argmin(np.abs(z - zi)))
    print(f" {z[j]:6.1f}| {xe_ab[j]:.5e}  {xe_ab[j]/xe_hy[j]-1:+.3e}   "
          f"{xe_rf[j]/xe_hy[j]-1:+.3e}     | {g_ab[j]/g_hy[j]-1:+.3e}    "
          f"{g_rf[j]/g_hy[j]-1:+.3e}")

win = (z > 900) & (z < 1300)
print(f"\n[summary] last-scattering window 900<z<1300:")
print(f"  ABCMB xe vs HyRec:   max|rel|={np.max(np.abs(xe_ab[win]/xe_hy[win]-1)):.3e}  "
      f"mean={np.mean(xe_ab[win]/xe_hy[win]-1):+.3e}")
print(f"  RecFast xe vs HyRec: max|rel|={np.max(np.abs(xe_rf[win]/xe_hy[win]-1)):.3e}  "
      f"mean={np.mean(xe_rf[win]/xe_hy[win]-1):+.3e}")
print(f"  ABCMB g vs HyRec:    max|rel|={np.max(np.abs(g_ab[win]/g_hy[win]-1)):.3e}")
print(f"  ABCMB g vs RecFast:  max|rel|={np.max(np.abs(g_ab[win]/g_rf[win]-1)):.3e}")
lo = node_ells[(node_ells >= 3) & (node_ells < 100)]
def mx(ref):
    e = [abs(bb_ab[np.where(node_ells==L)[0][0]]/ref[L]-1) for L in lo]
    return max(e)
print(f"  BB low-l max |ABCMB/HyRec-1|   = {mx(bb_hy):.3e}")
print(f"  BB low-l max |ABCMB/RecFast-1| = {mx(bb_rf):.3e}")
print("DONE", flush=True)
