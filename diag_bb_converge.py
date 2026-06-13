"""Tensor-sector grid convergence A/B at fixed ell nodes.

One BG + recomb solve, then tensor BB at a few ell nodes for:
  A. baseline           (Nlna_ten=500, perturbation k grid = scalar truncation)
  B. dense time         (Nlna_ten=2000)
  C. dense time + k     (Nlna_ten=2000, perturbation grid = transfer grid,
                         i.e. no source k-interpolation error)
"""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from abcmb.main import Model
from abcmb import tensors, spectrum
sys.path.insert(0, file_dir + '/pytests')
from accuracy_test_bb import PARAMS

model = Model(l_max=2500, lensing=False, tensors=True, l_max_g=12, l_max_pol_g=10)
params = model.add_derived_parameters(PARAMS)

def _to_float(v):
    arr = jnp.asarray(v)
    return arr.astype(jnp.float64) if arr.dtype.kind in 'iub' else arr
params = jax.tree_util.tree_map(_to_float, params)
pre_BG = model.get_BG_pre_recomb(params)
cpu = jax.devices('cpu')[0]
recomb_output = eqx.filter_jit(model.RecModel, backend='cpu')(
    (jax.device_put(pre_BG.recomb_inputs, cpu), jax.device_put(params, cpu)))
recomb_output = jax.tree_util.tree_map(_to_float, recomb_output)
BG = model.get_BG(params, pre_BG, recomb_output)

bessel_l_tab = np.asarray(spectrum.bessel_l_tab)
node_idxs = [int(np.argmin(np.abs(bessel_l_tab - l))) for l in [90, 237, 296, 450, 502]]

configs = {
    "A_base":        (dict(model.specs), model.TPE.k_axis_tensor),
    "B_dense_t":     (dict(model.specs, Nlna_ten=2000), model.TPE.k_axis_tensor),
    "C_dense_tk":    (dict(model.specs, Nlna_ten=2000), model.TSS.k_axis_transfer),
}

for name, (specs, k_axis) in configs.items():
    TPE = tensors.TensorPerturbationEvolver(
        model.species_list, model.species_dict, k_axis, specs,
        adjoint=model.TPE.adjoint)
    TPT = eqx.filter_jit(TPE.full_evolution)((BG, params))
    vals = []
    for idx in node_idxs:
        tt, te, ee, bb = eqx.filter_jit(model.TSS.Cl_one_ell)(idx, TPT, BG, params)
        vals.append((int(bessel_l_tab[idx]), float(bb), float(ee)))
    print(name)
    for l, bb, ee in vals:
        print(f"  l={l:4d}  BB={bb:.6e}  EE_ten={ee:.6e}")
print("DONE")
