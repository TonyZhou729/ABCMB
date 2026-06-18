"""Forward-call timer with tensors on vs off (GPU). Mirrors time_tests.py."""
import sys
sys.path.append('../')

import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)
from abcmb.main import Model

import time

params = {'r': 0.1}

for tensors in (False, True):
    specs = {
        "output_Cl": True,
        "output_Pk": True,
        "lensing": True,
        "tensors": tensors,
    }
    model = Model(**specs)
    p = params if tensors else {}
    for i in range(2):
        start = time.time()
        out = model(p)
        out.ClBB.block_until_ready()
        print(f"tensors={tensors} run {i}: {time.time()-start:.2f} s, "
              f"ClBB[l=2] = {out.ClBB[0]:.4e}, ClBB[78] = {out.ClBB[78]:.4e}")
