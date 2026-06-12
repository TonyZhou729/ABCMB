"""
End-to-end consistency check at omega_k = 0: the curved spectrum path
(curvature=True, exact hyperspherical recurrence at every ell) against the
flat path (Bessel tables + sparse-ell spline). Differences are bounded by the
flat path's table-interpolation + ell-spline error, so agreement at the
few-x-1e-4 level validates the wiring of the curved path.

Run on a GPU node:  python notes_curvature/test_flat_vs_curved_path.py
Runtime: ~2 model compiles + 4 evaluations; expect ~5-10 min total.
"""
import os, sys, time
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + "/..")

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
import numpy as np
from abcmb.main import Model

params = {
    'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225,
    'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4, 'N_nu_massive': 0,
    "tau_reion": 0.0544, "Delta_z_reion": 0.5, "z_reion_He": 3.5,
    "Delta_z_reion_He": 0.5, "exp_reion": 1.5,
}

if len(sys.argv) > 1:
    cases = (sys.argv[1] == "lensed",)
else:
    cases = (False, True)

for lensing in cases:
    print(f"=== lensing={lensing} ===")
    m_flat = Model(l_max=2500, lensing=lensing)
    m_curv = Model(l_max=2500, lensing=lensing, curvature=True)

    t0 = time.time(); out_f = m_flat(params); jax.block_until_ready(out_f.ClTT)
    t1 = time.time(); out_f = m_flat(params); jax.block_until_ready(out_f.ClTT)
    t2 = time.time()
    print(f"flat : compile+run {t1-t0:.1f} s, warm {t2-t1:.2f} s")

    t0 = time.time(); out_c = m_curv(params); jax.block_until_ready(out_c.ClTT)
    t1 = time.time(); out_c = m_curv(params); jax.block_until_ready(out_c.ClTT)
    t2 = time.time()
    print(f"curv : compile+run {t1-t0:.1f} s, warm {t2-t1:.2f} s")

    ells = np.asarray(out_f.l)
    for nm in ("ClTT", "ClEE", "ClTE"):
        a = np.asarray(getattr(out_f, nm))
        b = np.asarray(getattr(out_c, nm))
        scale = np.abs(a) if nm != "ClTE" else np.sqrt(
            np.asarray(out_f.ClTT)*np.asarray(out_f.ClEE))
        rel = np.abs(b-a)/scale
        for lo, hi in ((2, 29), (30, 800), (801, 2500)):
            band = (ells >= lo) & (ells <= hi)
            print(f"  {nm} rel diff, ell {lo:4d}-{hi:4d}: max {rel[band].max():.3e}")
    pk_f, pk_c = np.asarray(out_f.Pk), np.asarray(out_c.Pk)
    print(f"  Pk rel diff (same path, sanity): {np.abs(pk_c-pk_f).max()/np.abs(pk_f).max():.3e}")
print("done")
