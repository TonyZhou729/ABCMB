"""Characterize the low-l raw-BB node residual (l=10 hit 4.2e-3 vs hp CLASS).

Is it a real transfer error or a reion-bump near-zero-crossing relative-error
artifact? Print, node by node: ABCMB C_l, hp-CLASS C_l, l(l+1)C_l/2pi (to see
the bump/trough shape), and rel err. Also confirm the recomb-region nodes
(l>=30) are all sub-2.5e-3.
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
from abcmb.main import Model
from abcmb.spectrum import bessel_l_tab
from accuracy_test_bb import PARAMS, ELLMIN, ELLMAX, class_tensor_hp_reference

model = Model(l_max=ELLMAX, lensing=False, tensors=True, l_max_g=12, l_max_pol_g=10)
abcmb_bb = np.asarray(model(PARAMS).ClBB)        # integer ells ELLMIN..ELLMAX
print("ABCMB done", flush=True)

cl_hp = class_tensor_hp_reference()
bb_hp = cl_hp["bb"]                              # indexed by ell, up to 500
print("CLASS hp done", flush=True)

nodes = np.asarray(bessel_l_tab)
nodes = nodes[(nodes >= ELLMIN) & (nodes <= 490)]

def row(L):
    a = abcmb_bb[L - ELLMIN]
    c = bb_hp[L]
    err = abs(a - c) / abs(c) if c != 0 else float("nan")
    dl = L * (L + 1) / (2 * np.pi)
    print(f"  l={L:4d} | ABCMB Cl={a: .4e} | hp Cl={c: .4e} | "
          f"l(l+1)Cl/2pi: ABCMB={a*dl: .4e} hp={c*dl: .4e} | rel={err:.3e}")

print("\n--- LOW-L NODES (reion bump / trough region) ---")
for L in nodes[nodes <= 60]:
    row(L)

print("\n--- RECOMB-REGION NODES (l in [60,490]) ---")
recomb = nodes[(nodes >= 60) & (nodes <= 490)]
maxerr, maxl = 0.0, None
for L in recomb:
    a, c = abcmb_bb[L - ELLMIN], bb_hp[L]
    e = abs(a - c) / abs(c)
    if e > maxerr:
        maxerr, maxl = e, L
    row(L)
print(f"\nmax node rel err in [60,490]: {maxerr:.3e} at l={maxl}")

lo = nodes[nodes <= 60]
e_lo = np.array([abs(abcmb_bb[L-ELLMIN]-bb_hp[L])/abs(bb_hp[L]) for L in lo])
print(f"max node rel err in [2,60]:   {e_lo.max():.3e} at l={lo[e_lo.argmax()]}")
print("DONE")
