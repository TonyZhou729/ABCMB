"""
Reverse-AD smoke test through the CURVED spectrum path.

jax.grad of sum(ClTT^2)-style losses w.r.t. (h, omega_b, omega_cdm, omega_k)
at Omega_k = +0.01, with Model(curvature=True, adjoint=RecursiveCheckpoint).
Checks grads are finite and reports peak GPU memory + wall-clock. The curved
lax.scan over ells is chunked with jax.checkpoint, so reverse residency should
be ~(n_chunks x carry) ~ 1 GiB-level on top of the PE reverse cost.

Run on GPU:  python notes_curvature/test_reverse_ad_curved.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys, time
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + "/..")

import numpy as np
import jax
import jax.numpy as jnp
import diffrax

print(f"JAX backend: {jax.default_backend()}")
from abcmb.main import Model

h = 0.6762
omega_k = 0.01*h**2
base = {
    'h': h, 'omega_cdm': 0.1193, 'omega_b': 0.0225,
    'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4, 'N_nu_massive': 0,
    "tau_reion": 0.0544, "Delta_z_reion": 0.5, "z_reion_He": 3.5,
    "Delta_z_reion_He": 0.5, "exp_reion": 1.5,
    'omega_k': omega_k,
}

model = Model(l_max=2500, lensing=False, curvature=True, omega_k_ref=omega_k,
              adjoint=diffrax.RecursiveCheckpointAdjoint)

grad_keys = ("h", "omega_b", "omega_cdm", "omega_k")

def loss(vals):
    p = dict(base)
    for k, v in zip(grad_keys, vals):
        p[k] = v
    out = model(p)
    return jnp.sum(out.ClTT**2)*1.e18

vals0 = jnp.array([base[k] for k in grad_keys])

t0 = time.time()
l0 = loss(vals0)
jax.block_until_ready(l0)
print(f"forward (compile+run): {time.time()-t0:.0f} s, loss = {float(l0):.6e}")

gfun = jax.grad(loss)
t0 = time.time()
g = gfun(vals0)
jax.block_until_ready(g)
t1 = time.time()
g2 = gfun(vals0)
jax.block_until_ready(g2)
t2 = time.time()
print(f"grad: compile+run {t1-t0:.0f} s, warm {t2-t1:.1f} s")
try:
    stats = jax.devices()[0].memory_stats()
    print(f"peak GPU mem: {stats['peak_bytes_in_use']/2**30:.2f} GiB")
except Exception:
    pass

gnp = np.asarray(g)
for k, v in zip(grad_keys, gnp):
    print(f"  dloss/d{k} = {v:.6e}")
assert np.all(np.isfinite(gnp)), "NON-FINITE GRADS"
print("ALL FINITE")
