"""
Tests of the log-spaced high-k extension of the perturbation k-grid (k_tail_max) and of the
log-log interpolation of P(k) across it.
"""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
file_dir = os.path.dirname(__file__)

import sys
sys.path.append(file_dir + '/../')

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp
import numpy as np

from abcmb import model_specs
from abcmb import ABCMBTools as tools


def test_loglog_interp_power_law():
    """Log-log interpolation is exact for a power law inside the table and clamps outside."""
    k_tab = jnp.geomspace(1e-3, 1., 200)
    P_tab = 3. * k_tab**(-2.3)
    k = jnp.geomspace(1e-3, 1., 50)
    P = tools.loglog_interp(k, k_tab, P_tab)
    assert np.allclose(np.asarray(P), 3. * np.asarray(k)**(-2.3), rtol=1e-10)
    P_out = tools.loglog_interp(jnp.array([1e-5, 100.]), k_tab, P_tab)
    assert np.allclose(np.asarray(P_out), [float(P_tab[0]), float(P_tab[-1])], rtol=1e-12)


def test_k_grid_tail():
    """k_tail_max > 0 continues the log-spaced extension of the perturbation k-grid up to k_tail_max;
    the CMB part of the grid, k_size_cmb and the P(k) output grid are unchanged."""
    for lensing in (False, True):
        specs_off = model_specs.load_specs({"lensing": lensing})
        specs_on = model_specs.load_specs({"lensing": lensing, "k_tail_max": 100.})
        k_off, kout_off = model_specs.get_k_axis_perturbations(specs_off)
        k_on, kout_on = model_specs.get_k_axis_perturbations(specs_on)
        assert specs_on["k_size_cmb"] == specs_off["k_size_cmb"]
        assert specs_on["k_max_pert"] == specs_off["k_max_pert"]
        assert k_on.shape[0] > k_off.shape[0]
        assert np.allclose(np.asarray(kout_off), np.asarray(kout_on))
        assert np.allclose(np.asarray(k_off), np.asarray(k_on)[:k_off.shape[0]])
        assert float(k_on[-1]) >= specs_on["k_tail_max"]
        assert np.all(np.diff(np.asarray(k_on)) > 0.)
        ext = np.asarray(k_on)[np.asarray(k_on) >= specs_on["k_limber_start"]]
        assert ext[0] == specs_on["k_limber_start"]
        # at most k_per_decade_for_pk nodes per decade on the extension, more near the BAO scale
        assert np.all(np.diff(np.log10(ext)) <= 1. / specs_on["k_per_decade_for_pk"] + 1e-12)
        assert np.isclose(np.diff(np.log10(ext))[-1], 1. / specs_on["k_per_decade_for_pk"])


def test_k_grid_no_tail_by_default():
    """Without lensing and without k_tail_max the grid stops at the CMB range / k_max."""
    specs = model_specs.load_specs({"lensing": False})
    k, _ = model_specs.get_k_axis_perturbations(specs)
    assert not np.isfinite(specs["k_limber_start"])
    assert float(k[-1]) < 1.
