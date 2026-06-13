"""
A/B the closed-universe transfer grids at Omega_k = -0.01 (unlensed, CPU):
old plain k-walk vs integer-nu lattice. Dumps band maxima of the Cl
difference, locates the worst l, and compares the k-integrand at l = 137
between the grids (trapezoid-resolution diagnostic).
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + "/..")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from abcmb.main import Model

h = 0.6762
omega_k = -0.01*h**2
base = {
    'h': h, 'omega_cdm': 0.1193, 'omega_b': 0.0225,
    'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4, 'N_nu_massive': 0,
    "tau_reion": 0.0544, "Delta_z_reion": 0.5, "z_reion_He": 3.5,
    "Delta_z_reion_He": 0.5, "exp_reion": 1.5,
    'omega_k': omega_k,
}

outs = {}
for tag, lattice in (("kwalk", False), ("lattice", True)):
    m = Model(l_max=800, lensing=False, curvature=True, omega_k_ref=omega_k,
              closed_integer_nu=lattice)
    print(f"{tag}: Nk_transfer = {m.SS.k_axis_transfer.shape[0]}")
    out = m(base)
    jax.block_until_ready(out.ClTT)
    outs[tag] = (m, out)

ells = np.asarray(outs["kwalk"][1].l)
for nm in ("ClTT", "ClEE"):
    a = np.asarray(getattr(outs["kwalk"][1], nm))
    b = np.asarray(getattr(outs["lattice"][1], nm))
    rel = np.abs(b-a)/np.abs(a)
    for lo, hi in ((2, 29), (30, 200), (201, 500), (501, 800)):
        band = (ells >= lo) & (ells <= hi)
        am = ells[band][rel[band].argmax()]
        print(f"{nm} kwalk-vs-lattice, ell {lo:3d}-{hi:3d}: max {rel[band].max():.3e} at l={am}")

# k-integrand at l=137 for both grids
import abcmb.ABCMBTools as tools
for tag in ("kwalk", "lattice"):
    m, out = outs[tag]
    params = m.add_derived_parameters(base)
    sources = m.SS._transfer_sources(out.PT, out.BG, params)
    tt, te, ee = m.SS._Cl_all_ells_curved(sources, params)
    k = np.asarray(m.SS.k_axis_transfer)
    K = float(params['K'])
    nu = np.sqrt(k**2 + K)/np.sqrt(K)
    print(f"{tag}: Cl137(TT) = {float(np.asarray(tt)[135]):.6e}; "
          f"nu range [{nu[0]:.1f}, {nu[-1]:.1f}], Nk={len(k)}; "
          f"max dnu in nu<600: {np.diff(nu[nu<600]).max():.2f}; "
          f"min dnu: {np.diff(nu).min():.3f}")
print("done")
