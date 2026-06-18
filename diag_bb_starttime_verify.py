"""Verify the tensor start-time fix end-to-end: cap lna_start at -14 instead of
-10 (tensors.py:376). Confirm it (a) halves the low-l recomb-region excess,
(b) does NOT degrade high-l (122-490), (c) modest timing cost.

Replicates full_evolution + get_Cl at NODES (no spline confound) for the
baseline (-10 cap) and the -14 cap, vs CLASS-HyRec, across ALL tensor nodes.
"""
import os, time
os.environ["JAX_PLATFORM_NAME"] = "gpu"
import sys
file_dir = "/pscratch/sd/c/carag/ABCMB-bmodes"
sys.path.insert(0, file_dir); sys.path.insert(0, file_dir + "/pytests")
import jax
from jax import vmap
import diffrax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from abcmb.main import Model
from abcmb.spectrum import bessel_l_tab
from accuracy_test_bb import PARAMS, ELLMAX, class_tensor_hp_reference

bb_hp = class_tensor_hp_reference()["bb"]
print("CLASS hp done", flush=True)
model = Model(l_max=ELLMAX, lensing=False, tensors=True, l_max_g=12, l_max_pol_g=10)
out = model(PARAMS)
BG, params = out.BG, out.params
TPE, TSS = model.TPE, model.TSS
S = model.specs
node_ells = np.asarray(bessel_l_tab)[np.asarray(TSS.tensor_ells_indices)]
print("ABCMB build done", flush=True)


def evolve(k, lna_save, cap):
    lna_start = jnp.minimum(TPE.get_starting_time(k, (BG, params)), cap)
    y0 = TPE.initial_conditions_one_k(k, lna_start, (BG, params))
    ctrl = diffrax.PIDController(
        pcoeff=S["pcoeff_PE"], icoeff=S["icoeff_PE"], dcoeff=S["dcoeff_PE"],
        rtol=S.get("rtol_ten", 1.e-5), atol=S.get("atol_ten", 1.e-9))
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(TPE.get_derivatives), diffrax.Kvaerno5(),
        t0=lna_start, t1=0.0, dt0=1.e-2, y0=y0, stepsize_controller=ctrl,
        max_steps=8192, saveat=diffrax.SaveAt(ts=lna_save),
        args=(k, BG, params), adjoint=diffrax.ForwardMode())
    return sol.ys


def bb_nodes(cap):
    lna = jnp.linspace(BG.lna_transfer_start, 0., S.get("Nlna_ten", 500))
    res = vmap(lambda k: evolve(k, lna, cap))(TPE.k_axis_tensor)
    res = res.transpose(2, 1, 0)
    TPT = TPE.make_output_table(lna, res, (BG, params))
    o = vmap(TSS.Cl_one_ell, in_axes=(0, None, None, None))(
        TSS.tensor_ells_indices, TPT, BG, params)
    return np.asarray(jax.block_until_ready(o[3]))


# warm + time each cap
res = {}
for cap in (-10.0, -14.0):
    bb_nodes(cap)  # compile
    t0 = time.time()
    res[cap] = bb_nodes(cap)
    print(f"  cap={cap}: warm {time.time()-t0:.2f}s", flush=True)

print("\n--- BB rel err vs CLASS-HyRec, ALL nodes: cap=-10 (baseline) vs cap=-14 ---")
print("   l   |   cap=-10    |   cap=-14    | improved?")
for L in node_ells[node_ells <= 490]:
    i = np.where(node_ells == L)[0][0]
    c = bb_hp[L]
    e10 = (res[-10.0][i] - c) / c
    e14 = (res[-14.0][i] - c) / c
    flag = "yes" if abs(e14) < abs(e10) - 1e-5 else ("WORSE" if abs(e14) > abs(e10) + 1e-5 else "~same")
    print(f" {L:5d} | {e10:+.3e} | {e14:+.3e} | {flag}")

def band_max(cap, lo, hi):
    m = (node_ells >= lo) & (node_ells <= hi)
    e = [abs(res[cap][np.where(node_ells == L)[0][0]] - bb_hp[L]) / bb_hp[L]
         for L in node_ells[m]]
    return max(e)
print("\n[summary] max |rel err| vs CLASS-HyRec:")
for lo, hi in [(3, 99), (100, 490)]:
    print(f"  nodes {lo}-{hi}:  cap=-10 {band_max(-10.,lo,hi):.3e}  ->  "
          f"cap=-14 {band_max(-14.,lo,hi):.3e}")
print("DONE", flush=True)
