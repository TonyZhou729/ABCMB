"""
Branch-agnostic scalar forward timer (no `tensors` kwarg, so it runs on
5eabbab / origin/curvature / bmodes alike). Multi-warm min/median for
lensing=False and lensing=True. Used to measure the PURE curvature A/B:
origin/curvature (3174b25, recurrence) vs its merge-base 5eabbab (table+spline),
controlling for any bmodes-specific overhead.
"""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import abcmb
print("abcmb:", abcmb.__file__)
from abcmb.main import Model

NWARM = 6


def time_cfg(lensing, label):
    m = Model(l_max=2500, lensing=lensing)
    p = {}
    t = time.perf_counter(); o = m(p); o.ClTT.block_until_ready()
    print(f"  {label} compile+1st: {time.perf_counter()-t:6.2f} s")
    ts = []
    for _ in range(NWARM):
        t = time.perf_counter(); o = m(p); o.ClTT.block_until_ready()
        ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    print(f"  {label} warm: min {ts.min():.3f}s  median {np.median(ts):.3f}s  "
          f"runs {np.array2string(ts, precision=2, separator=',')}")
    return ts.min(), np.median(ts)


print("=== scalar forward timing (multi-warm) ===")
f_min, _ = time_cfg(False, "lensing=False")
t_min, _ = time_cfg(True,  "lensing=True ")
print(f"\nSUMMARY  lensing=False min {f_min:.3f}s | lensing=True min {t_min:.3f}s")
