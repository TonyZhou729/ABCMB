"""
Dual scalar-spectrum path sanity check (flat table vs curved recurrence).

The dual-path change gates the scalar get_Cl on the static SpectrumSolver.curvature
flag:
  - curvature=False (flat, the common path): fast sparse-ell tabulated-Bessel
    transfer + CubicSpline (origin/main's validated path).
  - curvature=True (omega_k != 0): the every-ell hyperspherical-Bessel
    recurrence (reduces to j_l at K=0).

Decisive cross-check: at omega_k = 0 the curved recurrence reduces to the flat
j_l, so the two branches must agree to the (sparse-ell CubicSpline) scalar
floor. Also runs a genuinely-curved config to confirm the curved branch still
produces finite, sensible Cls after the get_Cl restructure.

Cheap: 3 forward calls (1 compile + 1 warm each), no AD.
Run on GPU inside an allocation, e.g.:
  srun --jobid=<id> --overlap --ntasks=1 --cpus-per-task=32 \\
       python validate_dual_path.py
"""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import abcmb
print("abcmb:", abcmb.__file__)
from abcmb.main import Model


def run(curvature, omega_k, label):
    m = Model(l_max=2500, lensing=False, curvature=curvature, omega_k_ref=omega_k)
    p = {} if omega_k == 0. else {"omega_k": omega_k}
    o = m(p)
    o.ClTT.block_until_ready()
    tt, te, ee = np.asarray(o.ClTT), np.asarray(o.ClTE), np.asarray(o.ClEE)
    finite = np.all(np.isfinite(tt)) and np.all(np.isfinite(te)) and np.all(np.isfinite(ee))
    print(f"[{label}] curvature={curvature} omega_k={omega_k}  finite={finite}  "
          f"ClTT[2:5]={tt[:3]}")
    return o.l, tt, te, ee


print("\n=== flat (curvature=False, table path) ===")
l_flat, tt_flat, te_flat, ee_flat = run(False, 0., "flat-table")

print("\n=== curved branch at K=0 (curvature=True, omega_k=0, recurrence -> j_l) ===")
l_k0, tt_k0, te_k0, ee_k0 = run(True, 0., "curved-K0")

print("\n=== genuinely curved (curvature=True, omega_k=-0.05, closed) ===")
l_c, tt_c, te_c, ee_c = run(True, -0.05, "curved-0.05")

def relerr(a, b):
    m = np.abs(b) > 0
    return np.abs(a[m] - b[m]) / np.abs(b[m])

print("\n=== flat-table vs curved-K0 agreement (should be ~scalar spline floor) ===")
for name, a, b in [("TT", tt_flat, tt_k0), ("TE", te_flat, te_k0), ("EE", ee_flat, ee_k0)]:
    e = relerr(a, b)
    print(f"  {name}: max rel err = {e.max():.3e}  median = {np.median(e):.3e}")
