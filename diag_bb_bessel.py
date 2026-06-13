"""Isolate Bessel-table error in the tensor E/B radial functions.

Reruns the tensor pipeline, then for a few ells computes the B/E transfer
and Cl both with the production phi0/phi1/phi2 tables and with scipy's
exact spherical Bessel functions, on the same sources and grids.
"""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap
import numpy as np
from scipy.special import spherical_jn
from interpax import CubicSpline

from abcmb.main import Model
from abcmb import spectrum
sys.path.insert(0, file_dir + '/pytests')
from accuracy_test_bb import PARAMS

model = Model(l_max=2500, lensing=False, tensors=True, l_max_g=12, l_max_pol_g=10)

full_params = model.add_derived_parameters(PARAMS)
# Re-run pipeline up to BG + tensor sources (CPU)
import equinox as eqx

def _to_float(v):
    arr = jnp.asarray(v)
    if arr.dtype.kind in 'iub':
        return arr.astype(jnp.float64)
    return arr
params = jax.tree_util.tree_map(_to_float, full_params)
pre_BG = model.get_BG_pre_recomb(params)
cpu_dev = jax.devices('cpu')[0]
recomb_output = eqx.filter_jit(model.RecModel, backend='cpu')(
    (jax.device_put(pre_BG.recomb_inputs, cpu_dev), jax.device_put(params, cpu_dev)))
recomb_output = jax.tree_util.tree_map(_to_float, recomb_output)
BG = model.get_BG(params, pre_BG, recomb_output)
TPT = model.TPE.full_evolution((BG, params))

TSS = model.TSS
k_axis = np.asarray(TSS.k_axis_transfer)
lna_axis = np.asarray(TPT.lna[:-1])
delta_lna = float(TPT.lna[-1] - TPT.lna[-2])
tau0 = float(BG.tau0)
tau = np.asarray(BG.tau(jnp.array(lna_axis)))
aH = np.asarray(BG.aH(jnp.array(lna_axis), params))

interp_column = lambda col: CubicSpline(jnp.log10(TPT.k), col, check=False)(jnp.log10(jnp.array(k_axis)))
sourceT2 = np.asarray(vmap(interp_column)(TPT.source_T2[:-1, :]))
sourceE = np.asarray(vmap(interp_column)(TPT.source_E[:-1, :]))

w = np.full(len(lna_axis), delta_lna)
w[0] = 0.5 * delta_lna

bessel_l_tab = np.asarray(spectrum.bessel_l_tab)

Ph_over_k = 4. * np.pi * float(params['r']) * float(params['A_s']) \
    * (k_axis / TSS.k_pivot)**float(params['n_t']) / k_axis

np.savez(file_dir + "/diag_bb_bessel_inputs.npz",
         k_axis=k_axis, lna_axis=lna_axis, tau=tau, aH=aH, tau0=tau0,
         sourceT2=sourceT2, sourceE=sourceE, w=w, Ph_over_k=Ph_over_k)

for l in [231, 308, 437]:  # use actual node values if present
    # nearest node
    idx = int(np.argmin(np.abs(bessel_l_tab - l)))
    lnode = int(bessel_l_tab[idx])

    # --- production (table) path: call TSS.Cl_one_ell under jit, as the
    # pipeline does (module-level bessel tabs may live on GPU; jit closure
    # transfer handles them, eager mixing does not)
    tt_p, te_p, ee_p, bb_p = eqx.filter_jit(TSS.Cl_one_ell)(idx, TPT, BG, params)

    # --- exact scipy path
    x = (tau0 - tau)[:, None] * k_axis[None, :]   # (Nlna, Nk)
    j_l = spherical_jn(lnode, x)
    j_lp = spherical_jn(lnode, x, derivative=True)
    j_lpp = ((lnode * (lnode + 1) / x**2 - 1.) * j_l - 2. * j_lp / x)
    facT = np.sqrt(3. / 8. * (lnode + 2) * (lnode + 1) * lnode * (lnode - 1))
    radT = facT * j_l / x**2
    radE = 0.25 * (j_lpp + 4. * j_lp / x - (1. - 2. / x**2) * j_l)
    radB = 0.5 * (j_lp + 2. * j_l / x)

    DT = np.sum(w[:, None] * sourceT2 / aH[:, None] * radT, axis=0)
    DE = np.sum(w[:, None] * sourceE / aH[:, None] * radE, axis=0)
    DB = np.sum(w[:, None] * sourceE / aH[:, None] * radB, axis=0)

    tt_x = np.trapezoid(Ph_over_k * DT**2, k_axis)
    ee_x = np.trapezoid(Ph_over_k * DE**2, k_axis)
    bb_x = np.trapezoid(Ph_over_k * DB**2, k_axis)

    print(f"l={lnode}: BB table {float(bb_p):.6e}  exact {bb_x:.6e}  "
          f"rel {abs(float(bb_p)-bb_x)/bb_x:.4f}")
    print(f"         EE table {float(ee_p):.6e}  exact {ee_x:.6e}  "
          f"rel {abs(float(ee_p)-ee_x)/ee_x:.4f}")
    print(f"         TT table {float(tt_p):.6e}  exact {tt_x:.6e}  "
          f"rel {abs(float(tt_p)-tt_x)/tt_x:.4f}")
print("DONE")
