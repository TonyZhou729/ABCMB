"""Causal test: is the +0.4% low-l BB from UNDER-RESOLVING the tiny tight-coupling
photon tensor quadrupole?

diag_bb_tca_origin.py showed ABCMB's shear_g is FROZEN at a constant ~3e-12
through deep tight coupling (identical to 5 sig figs over z=4000->1600), while
CLASS's TCA quadrupole evolves physically. 3e-12 sits below atol_ten=1e-9 (and
below the ladder's tightest 1e-11), so the adaptive solver never resolves the
slow drift -> a +0.1% residual in the moments emerging from tight coupling ->
+0.4% low-l BB.

Force resolution of the tight-coupling era via a step-size cap dtmax (in lna)
on the tensor PIDController, and via much tighter atol. If low-l BB drops toward
CLASS-HyRec -> under-resolution confirmed AND fixable.

GPU: one model build + a few re-solves (dtmax/atol variants) + CLASS-HyRec ref.
~6 min warm.
"""
import os
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

bb_hp = class_tensor_hp_reference()["bb"]   # HyRec default reference
print("CLASS hp (HyRec) done", flush=True)

model = Model(l_max=ELLMAX, lensing=False, tensors=True, l_max_g=12, l_max_pol_g=10)
out = model(PARAMS)
BG, params = out.BG, out.params
TPE, TSS = model.TPE, model.TSS
S = model.specs
node_ells = np.asarray(bessel_l_tab)[np.asarray(TSS.tensor_ells_indices)]
print("ABCMB baseline done", flush=True)


def evolve(k, lna, args, dtmax, atol, rtol):
    lna_start = jnp.minimum(TPE.get_starting_time(k, args), -10.)
    y0 = TPE.initial_conditions_one_k(k, lna_start, args)
    ctrl = diffrax.PIDController(
        pcoeff=S["pcoeff_PE"], icoeff=S["icoeff_PE"], dcoeff=S["dcoeff_PE"],
        rtol=rtol, atol=atol, dtmax=dtmax)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(TPE.get_derivatives), diffrax.Kvaerno5(),
        t0=lna_start, t1=0.0, dt0=1.e-3, y0=y0, stepsize_controller=ctrl,
        max_steps=60000, saveat=diffrax.SaveAt(ts=lna),
        args=(k, *args), adjoint=diffrax.ForwardMode())
    return sol.ys


def bb_nodes(dtmax, atol, rtol):
    lna = jnp.linspace(BG.lna_transfer_start, 0., S.get("Nlna_ten", 500))
    res = vmap(lambda k: evolve(k, lna, (BG, params), dtmax, atol, rtol))(
        TPE.k_axis_tensor)
    res = res.transpose(2, 1, 0)
    TPT = TPE.make_output_table(lna, res, (BG, params))
    out = vmap(TSS.Cl_one_ell, in_axes=(0, None, None, None))(
        TSS.tensor_ells_indices, TPT, BG, params)
    return np.asarray(jax.block_until_ready(out[3]))


INF = jnp.inf
variants = [
    ("baseline (dtmax=inf, atol=1e-9)", INF, 1.e-9, 1.e-5),
    ("dtmax=0.02",                       0.02, 1.e-9, 1.e-5),
    ("dtmax=0.005",                      0.005, 1.e-9, 1.e-5),
    ("atol=1e-13",                        INF, 1.e-13, 1.e-6),
    ("dtmax=0.005 + atol=1e-13",         0.005, 1.e-13, 1.e-6),
]
res = {}
for name, dm, at, rt in variants:
    res[name] = bb_nodes(dm, at, rt)
    print(f"  {name} done", flush=True)

print("\n--- low-l BB rel err vs CLASS-HyRec (does forcing TC resolution help?) ---")
hdr = "  l   " + "".join(f"| {n.split('(')[0].strip()[:14]:>14s} " for n, *_ in variants)
print(hdr)
for L in node_ells[node_ells <= 130]:
    i = np.where(node_ells == L)[0][0]
    c = bb_hp[L]
    cells = "".join(f"| {(res[n][i]-c)/c:+12.3e} " for n, *_ in variants)
    print(f" {L:4d} {cells}")

lo = node_ells[(node_ells >= 3) & (node_ells < 100)]
def mx(arr):
    e = [abs(arr[np.where(node_ells == L)[0][0]] - bb_hp[L]) / bb_hp[L] for L in lo]
    return max(e), int(lo[int(np.argmax(e))])
print("\n[summary] max |rel err| low-l (3<=l<100) vs CLASS-HyRec:")
for n, *_ in variants:
    print(f"  {n:34s}: {mx(res[n])}")
print("DONE", flush=True)
