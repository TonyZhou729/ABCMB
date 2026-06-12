"""
Cheap CPU smoke test: construct flat/curved Models at small l_max and run the
full pipeline once each (flat omega_k=0; curved open and closed) to catch
syntax/trace/shape errors before spending GPU time. ~5-10 min CPU.
"""
import os, sys, time
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + "/..")

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
import numpy as np
from abcmb.main import Model

base = {
    'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225,
    'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4, 'N_nu_massive': 0,
    "tau_reion": 0.0544, "Delta_z_reion": 0.5, "z_reion_He": 3.5,
    "Delta_z_reion_He": 0.5, "exp_reion": 1.5,
}

def run(tag, model_kwargs, params):
    t0 = time.time()
    m = Model(**model_kwargs)
    out = m(params)
    jax.block_until_ready(out.ClTT)
    tt = np.asarray(out.ClTT)
    assert np.all(np.isfinite(tt)), f"{tag}: non-finite ClTT"
    assert np.all(np.isfinite(np.asarray(out.Pk))), f"{tag}: non-finite Pk"
    print(f"[OK] {tag}: {time.time()-t0:.0f} s, ClTT[l=10] = {tt[8]:.4e}")
    return out

kw = dict(l_max=250, lensing=False)
out_flat = run("flat   omega_k=0  (table path)", kw, base)
out_c0   = run("curved omega_k=0  (recurrence) ", dict(curvature=True, **kw), base)

h2 = base['h']**2
out_op = run("curved Omega_k=+0.05 (open)     ",
             dict(curvature=True, omega_k_ref=0.05*h2, **kw),
             dict(base, omega_k=0.05*h2))
out_cl = run("curved Omega_k=-0.05 (closed)   ",
             dict(curvature=True, omega_k_ref=-0.05*h2, **kw),
             dict(base, omega_k=-0.05*h2))

# flat-vs-curved at omega_k=0 quick consistency
a, b = np.asarray(out_flat.ClTT), np.asarray(out_c0.ClTT)
rel = np.abs(b-a)/np.abs(a)
print(f"flat vs curved path @K=0 (l_max=250): max rel {rel.max():.3e}")
# open/closed should bracket flat at low ell (sanity, not assert)
print("done")
