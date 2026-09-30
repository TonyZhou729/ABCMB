"""
Tests of abcmb.halofit (HALOFIT non-linear corrections).

The first tests are self-contained consistency checks of the module
on an analytic (BBKS) linear spectrum.  The last one feeds CLASS's own linear
spectrum through abcmb.halofit and compares with CLASS's HALOFIT output; it
is skipped if classy is not installed.
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
import pytest
from scipy.integrate import quad

from abcmb import halofit


def bbks_pk(k, A=9.e6, n_s=0.965, omega_m=0.143):
    """BBKS linear power spectrum, k in 1/Mpc, P in Mpc^3 (roughly Planck-like)."""
    q = k / omega_m
    T = np.log(1. + 2.34 * q) / (2.34 * q) * (
        1. + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4
    )**(-0.25)
    return A * k**n_s * T**2


def sigma2_dense(R):
    """Reference sigma^2(R) = int dlnk Delta^2 exp(-k^2 R^2) by adaptive quadrature."""
    f = lambda lnk: np.exp(lnk)**3 * bbks_pk(np.exp(lnk)) / (2. * np.pi**2) * np.exp(-(np.exp(lnk) * R)**2)
    # exp(-k^2 R^2) underflows to 0 at large k, which is exact; accuracy_test.py sets np.seterr(all='raise')
    # for the whole session.
    with np.errstate(under='ignore'):
        return quad(f, np.log(1e-6), np.log(1e4), limit=400, epsabs=0., epsrel=1e-10)[0]


def test_k_grid_halofit_tail():
    """nonlinear='halofit' switches the log-spaced high-k extension of the perturbation k-grid on
    even without lensing (the sigma(R) integrals run over it); the P(k) output grid still ends
    at k_max.  The extension itself is tested in k_tail_test.py."""
    from abcmb import model_specs
    specs_lin = model_specs.load_specs({"lensing": False})
    specs_nl = model_specs.load_specs({"lensing": False, "nonlinear": "halofit"})
    k_lin, kout_lin = model_specs.get_k_axis_perturbations(specs_lin)
    k_nl, kout_nl = model_specs.get_k_axis_perturbations(specs_nl)
    assert not np.isfinite(specs_lin["k_limber_start"]) and np.isfinite(specs_nl["k_limber_start"])
    assert specs_nl["k_size_cmb"] == specs_lin["k_size_cmb"]
    assert np.allclose(np.asarray(kout_lin), np.asarray(kout_nl))
    assert np.allclose(np.asarray(k_lin), np.asarray(k_nl)[:k_lin.shape[0]])
    assert float(k_nl[-1]) >= specs_nl["k_tail_max"]
    assert specs_nl["k_max"] == specs_lin["k_max"]


def test_halofit_parameters_truncated_table():
    """
    Even with a table ending at only 5 Mpc^-1 the
    HALOFIT parameters of a Planck-like spectrum agree with a much longer
    table at z=0 and at a growth-suppressed amplitude mimicking z~2.
    """
    lnk_full = jnp.linspace(np.log(1e-4), np.log(1e3), 561)
    lnk_5 = jnp.linspace(np.log(1e-4), np.log(5.), 400)
    for amp in (1., 0.15):
        D2_full = jnp.exp(lnk_full)**3 * bbks_pk(jnp.exp(lnk_full)) / (2. * jnp.pi**2) * amp
        D2_5 = jnp.exp(lnk_5)**3 * bbks_pk(jnp.exp(lnk_5)) / (2. * jnp.pi**2) * amp
        ref = halofit.halofit_parameters(lnk_full, D2_full)
        cut = halofit.halofit_parameters(lnk_5, D2_5)
        assert abs(float(cut[0]) / float(ref[0]) - 1.) < 2e-3, amp
        assert abs(float(cut[1]) - float(ref[1])) < 2e-3, amp
        assert abs(float(cut[2]) - float(ref[2])) < 5e-3, amp


def test_halofit_parameters_consistency():
    """sigma^2(1/k_sigma) = 1 and n_eff, C match numerical derivatives of ln sigma^2."""
    lnk = jnp.linspace(np.log(1e-4), np.log(1e3), 561)
    k = jnp.exp(lnk)
    Delta2 = k**3 * bbks_pk(k) / (2. * jnp.pi**2)

    k_sigma, n_eff, C = halofit.halofit_parameters(lnk, Delta2)
    k_sigma, n_eff, C = float(k_sigma), float(n_eff), float(C)
    R = 1. / k_sigma

    assert 0.1 < k_sigma < 1.0, k_sigma
    assert abs(sigma2_dense(R) - 1.) < 1e-6

    eps = 1e-3
    lns = lambda lnR: np.log(sigma2_dense(np.exp(lnR)))
    lnR = np.log(R)
    d1 = (lns(lnR + eps) - lns(lnR - eps)) / (2 * eps)
    d2 = (lns(lnR + eps) - 2 * lns(lnR) + lns(lnR - eps)) / eps**2
    assert abs((-3. - d1) - n_eff) < 1e-4
    assert abs(-d2 - C) < 1e-4


def test_halofit_pk_limits():
    """P_NL -> P_lin on large scales, exceeds it on small scales, both prescriptions."""
    lnk = jnp.linspace(np.log(1e-4), np.log(1e3), 561)
    kg = jnp.exp(lnk)
    Delta2 = kg**3 * bbks_pk(kg) / (2. * jnp.pi**2)
    k_sigma, n_eff, C = halofit.halofit_parameters(lnk, Delta2)

    k = jnp.geomspace(1e-3, 10., 60)
    Pk_lin = bbks_pk(k)
    for prescription in halofit.PRESCRIPTIONS:
        Pk_nl = halofit.halofit_pk(k, Pk_lin, k_sigma, n_eff, C, 0.31, 0.69, -1., prescription=prescription)
        ratio = np.asarray(Pk_nl / Pk_lin)
        assert np.all(np.isfinite(ratio))
        assert abs(ratio[0] - 1.) < 1e-2
        assert ratio[-1] > 1.5
        # the correction grows monotonically at k > 0.1 Mpc^-1
        sel = np.asarray(k) > 0.1
        assert np.all(np.diff(ratio[sel]) > 0.)

    with pytest.raises(NotImplementedError):
        halofit.halofit_pk(k, Pk_lin, k_sigma, n_eff, C, 0.31, 0.69, -1., prescription="bogus")


def test_table_coverage():
    """The correction is on while the table reaches well beyond k_sigma and off when it does
    not, in particular for the clamped root k_sigma = 1/R_MIN; smooth and monotonic between."""
    k_tab_max = 110.
    assert float(halofit.table_coverage(0.24, k_tab_max)) == 1.
    assert float(halofit.table_coverage(k_tab_max / halofit.COVERAGE_ON, k_tab_max)) == 1.
    assert float(halofit.table_coverage(k_tab_max / halofit.COVERAGE_OFF, k_tab_max)) == 0.
    assert float(halofit.table_coverage(1. / halofit.R_MIN, k_tab_max)) == 0.
    w = np.asarray(halofit.table_coverage(jnp.geomspace(30., 90., 50), k_tab_max))
    assert np.all(np.diff(w) <= 0.) and w[0] == 1. and w[-1] == 0.
    g = jax.grad(lambda ks: halofit.table_coverage(ks, k_tab_max))(1. / halofit.R_MIN)
    assert float(g) == 0.


def test_table_coverage_smooth():
    """The weight is C-infinity in ln k_sigma: its derivatives vanish at both ends of the ramp
    (a cubic smoothstep would leave a jump of the second derivative there)."""
    k_tab_max = 110.
    f = lambda lnks: halofit.table_coverage(jnp.exp(lnks), k_tab_max)
    d2 = jax.grad(jax.grad(f))
    d3 = jax.grad(d2)
    ramp = np.log(halofit.COVERAGE_ON / halofit.COVERAGE_OFF)
    for u in (0.01, 0.99):   # just inside either end
        lnks = np.log(k_tab_max / halofit.COVERAGE_OFF) - u * ramp
        assert abs(float(d2(lnks))) < 1e-20 and abs(float(d3(lnks))) < 1e-20


def test_halofit_high_redshift_bbks():
    """Without table_coverage the clamped root gives a spurious enhancement at high redshift:
    the weight must be 0 there whatever the fitting formula returns."""
    lnk = jnp.linspace(np.log(1e-4), np.log(110.), 485)
    k = jnp.exp(lnk)
    Delta2 = k**3 * bbks_pk(k) / (2. * jnp.pi**2) / 200.      # growth-suppressed, z ~ 13
    k_sigma, n_eff, C = halofit.halofit_parameters(lnk, Delta2)
    assert float(k_sigma) > 110. / halofit.COVERAGE_OFF
    assert float(halofit.table_coverage(k_sigma, 110.)) == 0.


@pytest.mark.parametrize("m_nu", [0., 0.2])
def test_halofit_vs_class_same_input(m_nu):
    """
    Feed CLASS's linear P(k, z) through abcmb.halofit and compare with CLASS's
    own HALOFIT (Takahashi + Bird) output, which isolates the fitting-formula
    port from any difference between Boltzmann solvers.  The m_nu = 0.2 eV case
    (f_nu = 0.015) tests the Bird et al. massive-neutrino terms, which change
    P_NL by 2.5% there.
    """
    Class = pytest.importorskip("classy").Class

    cp = {
        "output": "mPk",
        "non_linear": "halofit",
        "P_k_max_1/Mpc": 50.,
        "z_max_pk": 3.,
        "h": 0.6762, "omega_b": 0.0225, "omega_cdm": 0.1193,
        "A_s": 2.12424e-9, "n_s": 0.9709, "YHe": 0.245,
    }
    if m_nu > 0.:
        cp.update({"N_ur": 2.0308, "N_ncdm": 1, "m_ncdm": m_nu, "T_ncdm": 0.71611})
    else:
        cp["N_ur"] = 3.044
    cosmo = Class()
    cosmo.set(cp)
    cosmo.compute()
    f_nu = (cosmo.Omega_m() - cosmo.Omega_b() - cosmo.Omega0_cdm()) / cosmo.Omega_m() if m_nu > 0. else 0.

    # Linear spectrum from CLASS on a fine grid up to its k_max; the sigma(R)
    # integrals run over that table only, as in ABCMB.
    k_tab = np.geomspace(1e-4, 49., 400)
    lnk = jnp.linspace(np.log(k_tab[0]), np.log(k_tab[-1]), 461)
    kg = jnp.exp(lnk)
    k = np.geomspace(1e-2, 20., 40)

    H0 = cosmo.Hubble(0.)
    for z in (0., 1., 2.):
        Pk_tab = np.array([cosmo.pk_lin(kk, z) for kk in k_tab])
        Delta2 = halofit.loglog_interp(kg, k_tab, Pk_tab) * kg**3 / (2. * jnp.pi**2)
        k_sigma, n_eff, C = halofit.halofit_parameters(lnk, Delta2)

        # CLASS tabulates k_nl on its (coarse) time grid and interpolates linearly
        # in tau, which is off by up to ~1% between nodes; z=0 is a node.
        k_nl_class = cosmo.nonlinear_scale(np.array([z]), 1)[0]
        tol = 1e-3 if z == 0. else 2e-2
        assert abs(float(k_sigma) / k_nl_class - 1.) < tol, (z, float(k_sigma), k_nl_class)

        Om = cosmo.Om_m(z)
        Ode = cosmo.Omega_Lambda() * (H0 / cosmo.Hubble(z))**2
        Pk_lin = np.array([cosmo.pk_lin(kk, z) for kk in k])
        Pk_nl = halofit.halofit_pk(k, Pk_lin, k_sigma, n_eff, C, Om, Ode, -1., f_nu=f_nu, h=cp["h"])
        ratio_class = np.array([cosmo.pk(kk, z) for kk in k]) / Pk_lin
        ratio = np.asarray(Pk_nl) / Pk_lin
        assert np.max(np.abs(ratio / ratio_class - 1.)) < 2e-3, (z, ratio / ratio_class)

    cosmo.struct_cleanup()
    cosmo.empty()
