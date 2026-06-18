"""Is the low-l raw-BB excess a tensor-solver convergence (tolerance) bias?

User hint: a too-loose solver tol (rtol_large_k_PE in a past session) caused
similar l-dependent amplitude biases. The tensor solver uses a uniform
rtol_ten=1e-5 / atol_ten=1e-9; this tests whether tightening it removes the
~0.4% low-l excess vs hp-CLASS.

Ladder (rtol_ten, atol_ten, max_steps_ten):
  A = (1e-5, 1e-9,  4096)   current default
  B = (1e-6, 1e-10, 8192)
  C = (1e-7, 1e-11, 16384)

If the low-l excess shrinks A->B->C and converges -> default tol under-
converged -> tighten it (real fix). If flat -> not solver tol; structural.
GPU.
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

def run(rtol_ten, atol_ten, max_steps_ten):
    try:
        m = Model(l_max=ELLMAX, lensing=False, tensors=True,
                  l_max_g=12, l_max_pol_g=10, rtol_ten=rtol_ten,
                  atol_ten=atol_ten, max_steps_ten=max_steps_ten)
        bb = np.asarray(m(PARAMS).ClBB)
        print(f"ABCMB done (rtol={rtol_ten:.0e}, atol={atol_ten:.0e}, "
              f"max_steps={max_steps_ten})", flush=True)
        return bb
    except Exception as e:
        print(f"ABCMB FAILED (rtol={rtol_ten:.0e}, atol={atol_ten:.0e}): "
              f"{type(e).__name__}", flush=True)
        return None

bbA = run(1.e-5, 1.e-9,  4096)
bbB = run(1.e-6, 1.e-10, 8192)
bbC = run(1.e-7, 1.e-11, 16384)
bb_hp = class_tensor_hp_reference()["bb"]
print("CLASS hp done", flush=True)

nodes = np.asarray(bessel_l_tab)
nodes = nodes[(nodes >= ELLMIN) & (nodes <= 490)]
sel = nodes[(nodes <= 100) | np.isin(nodes, [152, 237, 331, 450, 490])]

def e(bb, L):
    return float("nan") if bb is None else bb[L-ELLMIN]/bb_hp[L]-1

print("\n  l   |  A/hp - 1  |  B/hp - 1  |  C/hp - 1")
for L in sel:
    print(f" {L:4d} | {e(bbA,L):+.3e} | {e(bbB,L):+.3e} | {e(bbC,L):+.3e}")

lo = nodes[nodes <= 100]
for tag, bb in [("A", bbA), ("B", bbB), ("C", bbC)]:
    if bb is None:
        continue
    err = np.array([abs(bb[L-ELLMIN]/bb_hp[L]-1) for L in lo])
    print(f"max |{tag}/hp - 1| over l<=100: {err.max():.3e} "
          f"at l={lo[err.argmax()]}")
if bbA is not None and bbC is not None:
    ac = np.array([abs(bbA[L-ELLMIN]/bbC[L-ELLMIN]-1) for L in lo])
    print(f"max |A/C - 1| over l<=100 (default-vs-tight): {ac.max():.3e} "
          f"at l={lo[ac.argmax()]}")
print("\nIf the excess shrinks A->C -> tighten the default tol. "
      "If flat -> structural. DONE")
