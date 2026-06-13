"""ABCMB-side ODE tolerance A/B for the tensor solve (BB at ell nodes)."""
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
node_idxs = [int(np.argmin(np.abs(bessel_l_tab - l))) for l in [237, 296, 450, 490]]

configs = {
    "A_base": dict(model.specs),
    "D_tight_tol": dict(model.specs,
                        rtol_small_k_PE=1.e-7, atol_small_k_PE=1.e-12,
                        rtol_large_k_PE=1.e-6, atol_large_k_PE=1.e-10,
                        max_steps_PE=8192),
}

for name, specs in configs.items():
    TPE = tensors.TensorPerturbationEvolver(
        model.species_list, model.species_dict, model.TPE.k_axis_tensor,
        specs, adjoint=model.TPE.adjoint)
    TPT = eqx.filter_jit(TPE.full_evolution)((BG, params))
    line = f"{name:12s}"
    for idx in node_idxs:
        tt, te, ee, bb = eqx.filter_jit(model.TSS.Cl_one_ell)(idx, TPT, BG, params)
        line += f"  l{int(bessel_l_tab[idx])}:{float(bb):.6e}"
    print(line)
print("DONE")
