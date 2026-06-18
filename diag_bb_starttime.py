"""Last untested structural difference: the tensor integration START TIME / IC
transient.

ABCMB starts every tensor mode at lna = min(get_starting_time, -10) <= -15
(z > 3e6) with zero moments (gw=1/sqrt6). CLASS starts later, per-mode. The Pi
bias is largest early (z=1400 +0.16%) and decays toward recomb (z=950 +0.08%)
-- the flavor of a start-epoch-dependent IC transient.

Force ABCMB's t0 to several values (same zero-moment IC) and measure low-l BB:
  - EARLIER starts (t0 <= -15): is the baseline start-converged?
  - LATER starts (t0 > -15, CLASS-like): does BB drop TOWARD CLASS?
    -> if yes, ABCMB's earlier start captures real early buildup CLASS misses
       (ABCMB more accurate); if BB rises/flat, start isn't the cause.

For t0 <= -15 keep the source/integral grid floor at -15 (clean, identical grid).
For t0 > -15 use floor=t0 (the dropped early source ~0 since g=0 there).

GPU: one model build + several re-solves. ~5 min warm.
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

bb_hp = class_tensor_hp_reference()["bb"]
print("CLASS hp done", flush=True)

model = Model(l_max=ELLMAX, lensing=False, tensors=True, l_max_g=12, l_max_pol_g=10)
out = model(PARAMS)
BG, params = out.BG, out.params
TPE, TSS = model.TPE, model.TSS
S = model.specs
node_ells = np.asarray(bessel_l_tab)[np.asarray(TSS.tensor_ells_indices)]
print("ABCMB baseline build done", flush=True)


def evolve(k, t0, lna_save):
    y0 = TPE.initial_conditions_one_k(k, t0, (BG, params))  # zero moments, gw=1/sqrt6
    ctrl = diffrax.PIDController(
        pcoeff=S["pcoeff_PE"], icoeff=S["icoeff_PE"], dcoeff=S["dcoeff_PE"],
        rtol=S.get("rtol_ten", 1.e-5), atol=S.get("atol_ten", 1.e-9))
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(TPE.get_derivatives), diffrax.Kvaerno5(),
        t0=t0, t1=0.0, dt0=1.e-2, y0=y0, stepsize_controller=ctrl,
        max_steps=8192, saveat=diffrax.SaveAt(ts=lna_save),
        args=(k, *args(BG, params)), adjoint=diffrax.ForwardMode())
    return sol.ys


def args(BG, params):
    return (BG, params)


def bb_nodes_fixed_t0(t0, floor):
    lna = jnp.linspace(floor, 0., S.get("Nlna_ten", 500))
    res = vmap(lambda k: evolve(k, t0, lna))(TPE.k_axis_tensor)
    res = res.transpose(2, 1, 0)
    TPT = TPE.make_output_table(lna, res, (BG, params))
    o = vmap(TSS.Cl_one_ell, in_axes=(0, None, None, None))(
        TSS.tensor_ells_indices, TPT, BG, params)
    return np.asarray(jax.block_until_ready(o[3]))


def bb_baseline():
    lna = jnp.linspace(BG.lna_transfer_start, 0., S.get("Nlna_ten", 500))
    res = vmap(TPE.evolution_one_k, in_axes=[0, None, None])(
        TPE.k_axis_tensor, lna, (BG, params))
    res = res.transpose(2, 1, 0)
    TPT = TPE.make_output_table(lna, res, (BG, params))
    o = vmap(TSS.Cl_one_ell, in_axes=(0, None, None, None))(
        TSS.tensor_ells_indices, TPT, BG, params)
    return np.asarray(jax.block_until_ready(o[3]))


floor15 = float(BG.lna_transfer_start)
print(f"lna_transfer_start = {floor15:.3f}", flush=True)
variants = [("baseline(per-k start)", None, None),
            ("t0=-22", -22.0, floor15),
            ("t0=-18", -18.0, floor15),
            ("t0=-16", -16.0, floor15),
            ("t0=-13", -13.0, -13.0),
            ("t0=-11", -11.0, -11.0)]
res = {}
for name, t0, fl in variants:
    res[name] = bb_baseline() if t0 is None else bb_nodes_fixed_t0(t0, fl)
    print(f"  {name} done", flush=True)

print("\n--- low-l BB rel err vs CLASS-HyRec at forced tensor start times ---")
print("  l   " + "".join(f"| {n[:13]:>13s} " for n, *_ in variants))
for L in node_ells[node_ells <= 130]:
    i = np.where(node_ells == L)[0][0]
    c = bb_hp[L]
    print(f" {L:4d} " + "".join(f"| {(res[n][i]-c)/c:+11.3e} " for n, *_ in variants))

lo = node_ells[(node_ells >= 3) & (node_ells < 100)]
def mx(a):
    e = [abs(a[np.where(node_ells == L)[0][0]] - bb_hp[L]) / bb_hp[L] for L in lo]
    return max(e), int(lo[int(np.argmax(e))])
print("\n[summary] max |rel err| low-l (3<=l<100) vs CLASS-HyRec:")
for n, *_ in variants:
    print(f"  {n:22s}: {mx(res[n])}")
print("DONE", flush=True)
