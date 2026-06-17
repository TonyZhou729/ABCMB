"""
Reverse-AD smoke test for the B-modes (tensors=True) path.

Mirrors test_reverse_ad.py but exercises the new tensor sector:
  - TensorPerturbationEvolver (GW + photon/neutrino tensor hierarchies)
  - TensorSpectrumSolver (tensor radial transfer -> TT/TE/EE/BB)
  - the lensed_Cls EE<->BB mixing 4-tuple (lensing=True only)

For each lensing flag it takes jax.grad (via eqx.filter_grad, per the Phase-2
CPU/GPU backend constraint) of a sum-of-squares ClBB loss w.r.t. the tensor
amplitude r, tensor tilt n_t, and the standard LCDM params (which flow through
the shared background / recomb / scalar-lensing path). Reports peak GPU memory,
wall-clock, and whether every grad is finite.

NOTE: jax_debug_nans is deliberately OFF. The backward pass has a known,
forward-safe 0*inf-through-where in BG/PE/HyRex that debug_nans flags as a
false positive (see memory project_bessel_recurrence_handoff); finiteness is
checked explicitly on the returned grads instead.

Run on GPU inside an allocation:
  srun --jobid=<id> --overlap --ntasks=1 --cpus-per-task=32 \\
       python test_reverse_ad_bb.py
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
# Force the local worktree abcmb (with tensors.py) ahead of any editable install.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import traceback

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

jax.config.update("jax_enable_x64", True)

print(f"JAX backend: {jax.default_backend()}")
print(f"Devices:     {jax.devices()}")

import abcmb
print(f"abcmb from:  {abcmb.__file__}")
import abcmb.tensors  # noqa: F401  (fail loudly if the worktree copy lacks tensors)
from abcmb.main import Model

R_TENSOR = 0.1

# Fiducial LCDM + tensors (matches pytests/accuracy_test_bb.py).
base_params = {
    'h':          0.6762,
    'omega_cdm':  0.1193,
    'omega_b':    0.0225,
    'A_s':        2.12424e-9,
    'n_s':        0.9709,
    'Neff':       3.044,
    'YHe':        0.245,
    'TCMB0':      2.34865418e-4,
    'N_nu_massive': 0,
    'T_nu_massive': 0.71611,
    'm_nu_massive': 0.06,
    'tau_reion':  0.0544,
    'Delta_z_reion': 0.5,
    'z_reion_He': 3.5,
    'Delta_z_reion_He': 0.5,
    'exp_reion':  1.5,
    'r':   R_TENSOR,
}
# Pass n_t explicitly so it is an independent leaf we can differentiate w.r.t.
base_params['n_t'] = -R_TENSOR / 8. * (2. - R_TENSOR / 8. - base_params['n_s'])

# r and n_t are the tensor-specific knobs; the rest share the scalar pipeline.
GRAD_KEYS = ("r", "n_t", "h", "omega_b", "omega_cdm", "A_s", "n_s")


def _block(x):
    for leaf in jax.tree_util.tree_leaves(x):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _peak_gib():
    return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / (1024 ** 3)


def run_config(lensing):
    tag = f"lensing={lensing}"
    print(f"\n{'='*64}\n  tensors=True, {tag}\n{'='*64}")

    model = Model(
        l_max=2500,
        lensing=lensing,
        tensors=True,
        l_max_g=12,
        l_max_pol_g=10,
        adjoint=diffrax.RecursiveCheckpointAdjoint,
    )
    full_params = model.add_derived_parameters(base_params)
    grad_vals = tuple(jnp.asarray(full_params[k], dtype=jnp.float64) for k in GRAD_KEYS)

    def loss_fn(gv):
        p = dict(full_params)
        for k, v in zip(GRAD_KEYS, gv):
            p[k] = v
        out = model.run_cosmology_abbr(p)
        # ClBB is the discriminating tensor observable (plus lensing E->B).
        return jnp.sum(out.ClBB ** 2)

    # ---- forward ----
    t0 = time.perf_counter()
    lv = loss_fn(grad_vals)
    _block(lv)
    print(f"  fwd compile+1st : {time.perf_counter()-t0:7.2f} s")
    t0 = time.perf_counter()
    lv = loss_fn(grad_vals)
    _block(lv)
    print(f"  fwd warm        : {(time.perf_counter()-t0)*1e3:7.1f} ms")
    print(f"  loss (ClBB^2)   : {float(lv):+.6e}")
    print(f"  peak GPU mem    : {_peak_gib():7.3f} GiB (cumulative)")

    # ---- reverse-mode grad ----
    print(f"\n  -- eqx.filter_grad(loss_fn), {tag} --")
    grad_fn = eqx.filter_grad(loss_fn)
    try:
        t0 = time.perf_counter()
        g = grad_fn(grad_vals)
        _block(g)
        print(f"  rev compile+1st : {time.perf_counter()-t0:7.2f} s")
        t0 = time.perf_counter()
        g = grad_fn(grad_vals)
        _block(g)
        print(f"  rev warm        : {time.perf_counter()-t0:7.2f} s")
        print(f"  peak GPU mem    : {_peak_gib():7.3f} GiB (cumulative)")

        print("\n  grads (d ClBB^2 / d param):")
        all_finite = True
        for k, v in zip(GRAD_KEYS, g):
            vv = float(np.asarray(v))
            fin = np.isfinite(vv)
            all_finite &= fin
            print(f"    {k:12s} : {vv:+.6e}   ({'finite' if fin else 'NON-FINITE'})")
        if all_finite:
            print(f"\n  PASS ({tag}): all reverse-mode grads finite.")
        else:
            print(f"\n  WARN ({tag}): some grads non-finite.")
        return all_finite
    except Exception as e:
        print(f"\n  FAIL ({tag}): reverse-mode grad raised an exception.")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"\n  peak GPU mem at crash : {_peak_gib():7.3f} GiB")
        return False


if __name__ == "__main__":
    results = {}
    for lensing in (False, True):
        results[lensing] = run_config(lensing)
    print(f"\n{'='*64}\n  SUMMARY\n{'='*64}")
    for lensing, ok in results.items():
        print(f"  tensors=True, lensing={lensing:<5} : "
              f"{'PASS (finite)' if ok else 'FAIL'}")
