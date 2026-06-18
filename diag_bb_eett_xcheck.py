"""Localize the recomb-region tensor-BB excess: SOURCE vs RADIAL vs h.

Grid convergence (transfer + source) and reionization are RULED OUT. The
+0.4% excess lives in the pure-recombination BB. BB uses only the
polarization source S_E = sqrt(6) g Pi with radial function radB. Tensor EE
uses the SAME S_E but radial function radE; tensor TT adds the -hdot e^{-k}
GW-amplitude term with radT. So:

  excess in EE ~ excess in BB  ->  bias is in the SOURCE g*Pi (shared E/B)
  EE clean, BB high            ->  bias is in radB specifically
  TT, EE, BB all ~equally high ->  bias is in h (GW amplitude / normalization)

Compare ABCMB tensor TT/EE/BB at the spline nodes vs high-precision
tensor-only CLASS. One PE solve, cheap re-integration. GPU, ~3 min warm.
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
import jax.numpy as jnp
from abcmb.main import Model
from abcmb.spectrum import bessel_l_tab
from accuracy_test_bb import PARAMS, ELLMIN, ELLMAX, class_tensor_hp_reference

cl_hp = class_tensor_hp_reference()
print("CLASS hp done", flush=True)

model = Model(l_max=ELLMAX, lensing=False, tensors=True,
              l_max_g=12, l_max_pol_g=10)
output = model(PARAMS)
BG, params = output.BG, output.params
TPT = jax.block_until_ready(model.TPE.full_evolution((BG, params)))
print("ABCMB TPT done", flush=True)

# raw (un-splined) tensor spectra at the nodes
TSS = model.TSS
tt, te, ee, bb = jax.vmap(TSS.Cl_one_ell, in_axes=(0, None, None, None))(
    TSS.tensor_ells_indices, TPT, BG, params)
tt = np.asarray(tt); ee = np.asarray(ee); bb = np.asarray(bb)
node_ells = np.asarray(bessel_l_tab)[np.asarray(TSS.tensor_ells_indices)]

# CLASS-hp raw_cl is C_l (already the same convention as ABCMB get_Cl)
tt_hp = cl_hp["tt"]; ee_hp = cl_hp["ee"]; bb_hp = cl_hp["bb"]


def rel(a, c):
    return (a - c) / c if c != 0 else float("nan")


print("\n--- tensor TT / EE / BB rel err vs CLASS-hp, at nodes ---")
print("  l   |  TT err     EE err     BB err   | "
      "EE~BB? (source)  TT~BB? (h)")
for i, L in enumerate(node_ells):
    if L > 200:
        break
    et = rel(tt[i], tt_hp[L]); ee_e = rel(ee[i], ee_hp[L]); eb = rel(bb[i], bb_hp[L])
    src = "EE~BB" if abs(ee_e - eb) < 0.3 * abs(eb) else ""
    hh = "TT~BB" if abs(et - eb) < 0.3 * abs(eb) else ""
    print(f" {L:4d} | {et:+.3e}  {ee_e:+.3e}  {eb:+.3e} |  {src:8s}        {hh}")

# focused summary over the clean recomb-only band (12 <= l <= 120, reion ~0)
band = node_ells[(node_ells >= 12) & (node_ells <= 120)]
def bandmean(arr, hp):
    return np.mean([rel(arr[np.where(node_ells == L)[0][0]], hp[L]) for L in band])
print("\n[summary] mean rel err over recomb-only band 12<=l<=120:")
print(f"  TT: {bandmean(tt, tt_hp):+.3e}")
print(f"  EE: {bandmean(ee, ee_hp):+.3e}")
print(f"  BB: {bandmean(bb, bb_hp):+.3e}")
print("DONE", flush=True)
