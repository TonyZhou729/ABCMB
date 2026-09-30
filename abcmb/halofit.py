"""
HALOFIT non-linear corrections to the matter power spectrum.

This module is adapted from the JAX implementation of HALOFIT in jax-cosmo
(``jax_cosmo/power.py``, https://github.com/DifferentiableUniverseInitiative/jax_cosmo),
which is distributed under the MIT License reproduced below.  The fitting
formulae of Smith et al. (2003, astro-ph/0207664) and Takahashi et al.
(2012, arXiv:1208.2701) are ported from jax-cosmo.  Changes made for
ABCMB are:

* the linear power spectrum is ABCMB's tabulated P(k, z) rather than an
  analytic transfer function, and the sigma(R) integrals run over the
  tabulated k-range only, without extrapolation.  ``Model`` gives the
  perturbation k-grid a coarse log-spaced tail up to ``k_tail_max``
  (100 Mpc^{-1} by default, as CLASS/ACT use) so that the integrals are
  converged at all redshifts that matter;
* the non-linear scale is found by interpolating sigma^2(R) on a log-R grid
  (as in jax-cosmo) and then refined with a few Newton steps, and the
  effective index and curvature are normalised by the actual sigma^2(R) as in
  CLASS (identical to jax-cosmo when sigma^2(R_sigma) = 1 exactly);
* the massive-neutrino terms of Bird, Viel & Haehnelt (2012, arXiv:1109.4416)
  as implemented in CLASS/CAMB are included in the Takahashi prescription.
  They are proportional to f_nu and vanish exactly for cosmologies without
  massive neutrinos, in which case the formula is identical to jax-cosmo's;
* at high redshift the non-linear scale moves out of the tabulated k-range.  
  CLASS switches the correction off there.  jax-cosmo uses an analytical
  Pk_lin and integrates it to very large k. Here, since we use the real 
  numerical P_lin, we cannot integrate P_lin to k = 1e4 and follow 
  jax-cosmo, and we cannot shut the correction offabruptly as in CLASS 
  without breaking differentiability.  So``table_coverage`` gradually shuts 
  the correction off.

All wavenumbers are in Mpc^{-1} and all scales in Mpc.  The fitting formulae
depend only on the dimensionless Delta^2(k) = k^3 P(k)/(2 pi^2) and on
y = k / k_sigma, so they are independent of the choice of length unit.

----------------------------------------------------------------------------
MIT License

Copyright (c) 2022 Differentiable Universe Initiative
Copyright (c) 2026 Zilu Zhou, Cara Giovanetti, Hongwan Liu (modifications)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
----------------------------------------------------------------------------
"""
import numpy as np
import jax.numpy as jnp
from jax import config
from jax.nn import sigmoid
from .ABCMBTools import loglog_interp

config.update("jax_enable_x64", True)

__all__ = ["loglog_interp", "halofit_parameters", "halofit_pk", "table_coverage", "PRESCRIPTIONS"]

PRESCRIPTIONS = ("takahashi2012", "smith2003")

# Bracket (in Mpc) for the non-linear scale R_sigma, defined by sigma^2(R_sigma) = 1
# with a Gaussian filter, where sigma is the RMS of the linear density field.
# For any cosmology of interest, R_sigma(z=0) is a few Mpc, but at high redshift the 
# root may fall below R_MIN, in which case: R_sigma is set to R_MIN, 
# k_sigma = 1/R_MIN is large, and the non-linear correction vanishes.
R_MIN = 1.e-3
R_MAX = 1.e2
LNR_GRID = jnp.linspace(np.log(R_MIN), np.log(R_MAX), 51)

# Number of Newton refinements of ln R_sigma after the grid interpolation.  The
# iteration converges quadratically from a ~1e-3 initial error, so 3 is plenty.
N_NEWTON = 3

# The non-linear correction is applied in full while PT.k extends to at least
# COVERAGE_ON * k_sigma.  The correction is not applied below 
# COVERAGE_OFF * k_sigma (see table_coverage; fitting formula is spurious enhancement 
# here). Gap between them is a smooth off-ramp.
COVERAGE_ON = 2.5
COVERAGE_OFF = 1.5


def table_coverage(k_sigma, k_tab_max):
    """
    Weight of the non-linear correction given how far the table reaches beyond k_sigma.

    Parameters:
    -----------
    k_sigma : float
        Non-linear wavenumber from ``halofit_parameters`` (units: Mpc^{-1})
    k_tab_max : float
        Largest tabulated wavenumber entering the sigma(R) integrals (units: Mpc^{-1})

    Returns:
    --------
    float
        1 for k_tab_max >= COVERAGE_ON * k_sigma, 0 for k_tab_max <= COVERAGE_OFF * k_sigma
        (which includes the clamped root k_sigma = 1/R_MIN), smooth step in ln k between.

    """
    u = jnp.log(k_tab_max / k_sigma / COVERAGE_OFF) / np.log(COVERAGE_ON / COVERAGE_OFF)
    # Within 1e-3 of either end the step is exactly 0 or 1 in double precision (exp(-1/u)
    # underflows); evaluating it only inside avoids 0 * inf in its derivatives as u -> 0.
    inside = (u > 1.e-3) & (u < 1. - 1.e-3)
    u_in = jnp.where(inside, u, 0.5)
    return jnp.where(inside, sigmoid(1. / (1. - u_in) - 1. / u_in), jnp.where(u < 0.5, 0., 1.))


def _simpson_weights(n):
    """
    Composite Simpson weights for n uniformly spaced points (n odd).  Falls
    back to the trapezoid rule if n is even.  The step size is not included.
    """
    if n % 2 == 1:
        w = np.full(n, 2. / 3.)
        w[1::2] = 4. / 3.
        w[0] = w[-1] = 1. / 3.
    else:
        w = np.ones(n)
        w[0] = w[-1] = 0.5
    return jnp.asarray(w)


def halofit_parameters(lnk, Delta2, lnR_grid=LNR_GRID, n_newton=N_NEWTON):
    """
    Non-linear scale, effective spectral index and spectral curvature that
    enter the HALOFIT fitting formulae.

    Parameters:
    -----------
    lnk : array
        Uniform grid in ln k on which Delta2 is tabulated (k in Mpc^{-1}).
    Delta2 : array
        Dimensionless linear power spectrum k^3 P_lin(k)/(2 pi^2) on lnk.
    lnR_grid : array, optional
        Increasing grid in ln R (R in Mpc) bracketing the non-linear scale.
    n_newton : int, optional
        Number of Newton refinements of the root (default: N_NEWTON).

    Returns:
    --------
    tuple
        (k_sigma, n_eff, C) with k_sigma = 1/R_sigma in Mpc^{-1}, where
        sigma^2(R_sigma) = 1, n_eff = -3 - d ln sigma^2 / d ln R and
        C = - d^2 ln sigma^2 / d ln R^2, both evaluated at R_sigma.

    Notes:
    ------
    Uses the Gaussian-filtered variance of Smith et al. (2003), Appendix C,

        sigma^2(R) = int d ln k  Delta^2(k) exp(-k^2 R^2),

    integrated with Simpson's rule on the lnk grid.
    """
    dlnk = lnk[1] - lnk[0]
    k2 = jnp.exp(2. * lnk)
    wD = _simpson_weights(lnk.shape[0]) * dlnk * Delta2  # weighted integrand, (Nk,)

    # sigma^2 on the R grid; flip ln sigma^2 for jnp.interp
    sig2_grid = wD @ jnp.exp(-jnp.outer(k2, jnp.exp(2. * lnR_grid)))
    lnR = jnp.interp(0., jnp.log(sig2_grid)[::-1], lnR_grid[::-1])

    # Newton refinement of f(ln R) = ln sigma^2(R) = 0, with
    # d ln sigma^2 / d ln R = -s2/s1 where s2 = int 2 (kR)^2 Delta^2 e^{-(kR)^2}.
    for _ in range(n_newton):
        x2 = k2 * jnp.exp(2. * lnR)
        e = jnp.exp(-x2)
        s1 = jnp.sum(wD * e)
        s2 = jnp.sum(wD * e * 2. * x2)
        lnR = jnp.clip(lnR + jnp.log(s1) * s1 / s2, lnR_grid[0], lnR_grid[-1])

    x2 = k2 * jnp.exp(2. * lnR)
    e = jnp.exp(-x2)
    s1 = jnp.sum(wD * e)
    s2 = jnp.sum(wD * e * 2. * x2)
    s3 = jnp.sum(wD * e * 4. * x2 * (1. - x2))

    d1 = -s2 / s1                            # d ln sigma^2 / d ln R
    d2 = -s2**2 / s1**2 - s3 / s1            # d^2 ln sigma^2 / d ln R^2

    k_sigma = jnp.exp(-lnR)
    n_eff = -3. - d1
    C = -d2

    return k_sigma, n_eff, C


def halofit_pk(k, Pk_lin, k_sigma, n_eff, C, Omega_m, Omega_de, w,
               f_nu=0., h=1., prescription="takahashi2012"):
    """
    HALOFIT non-linear matter power spectrum.

    Parameters:
    -----------
    k : float or array
        Wavenumber (units: Mpc^{-1})
    Pk_lin : float or array
        Linear matter power spectrum at k and at the redshift of interest
        (units: Mpc^3)
    k_sigma : float
        Non-linear scale 1/R_sigma at that redshift (units: Mpc^{-1})
    n_eff : float
        Effective spectral index at R_sigma
    C : float
        Spectral curvature at R_sigma
    Omega_m : float
        Matter density fraction Omega_m(a) at that redshift
    Omega_de : float
        Dark energy density fraction Omega_de(a) at that redshift
    w : float
        Dark energy equation of state at that redshift
    f_nu : float, optional
        Fraction of the matter density in massive neutrinos today,
        Omega_nu/Omega_m (default: 0).  Only used by 'takahashi2012'.
    h : float, optional
        Reduced Hubble constant, needed to convert k to h/Mpc in the
        massive-neutrino term (default: 1).  Only used when f_nu != 0.
    prescription : str, optional
        'takahashi2012' (default) or 'smith2003'

    Returns:
    --------
    float or array
        Non-linear matter power spectrum P_NL(k) (units: Mpc^3)

    Notes:
    ------
    Appendix C of Smith et al. (2003) with the coefficients of Takahashi et
    al. (2012) or the original ones, as in jax-cosmo.  Equation numbers below
    refer to Smith et al.  The f_nu terms follow Bird, Viel & Haehnelt (2012)
    as implemented in CLASS (external/Halofit/halofit.c) and CAMB.
    """
    if prescription not in PRESCRIPTIONS:
        raise NotImplementedError(
            f"halofit prescription must be one of {PRESCRIPTIONS}, got '{prescription}'"
        )

    n = n_eff
    om_m = Omega_m
    om_de = Omega_de
    frac = om_de / (1.0 - om_m)

    if prescription == "smith2003":
        # eq C9 to C18
        a_n = 10 ** (
            1.4861
            + 1.8369 * n
            + 1.6762 * n**2
            + 0.7940 * n**3
            + 0.1670 * n**4
            - 0.6206 * C
        )
        b_n = 10 ** (0.9463 + 0.9466 * n + 0.3084 * n**2 - 0.9400 * C)
        c_n = 10 ** (-0.2807 + 0.6669 * n + 0.3214 * n**2 - 0.0793 * C)
        gamma_n = 0.8649 + 0.2989 * n + 0.1631 * C
        alpha_n = 1.3884 + 0.3700 * n - 0.1452 * n**2
        beta_n = 0.8291 + 0.9854 * n + 0.3401 * n**2
        mu_n = 10 ** (-3.5442 + 0.1908 * n)
        nu_n = 10 ** (0.9585 + 1.2857 * n)
    else:
        a_n = 10 ** (
            1.5222
            + 2.8553 * n
            + 2.3706 * n**2
            + 0.9903 * n**3
            + 0.2250 * n**4
            - 0.6038 * C
            + 0.1749 * om_de * (1 + w)
        )
        b_n = 10 ** (
            -0.5642 + 0.5864 * n + 0.5716 * n**2 - 1.5474 * C + 0.2279 * om_de * (1 + w)
        )
        c_n = 10 ** (0.3698 + 2.0404 * n + 0.8161 * n**2 + 0.5869 * C)
        gamma_n = 0.1971 - 0.0843 * n + 0.8460 * C
        alpha_n = jnp.abs(6.0835 + 1.3373 * n - 0.1959 * n**2 - 5.5274 * C)
        beta_n = (
            2.0379
            - 0.7354 * n
            + 0.3157 * n**2
            + 1.2490 * n**3
            + 0.3980 * n**4
            - 0.1682 * C
            + f_nu * (1.081 + 0.395 * n**2)      # Bird et al. (2012)
        )
        mu_n = 0.0
        nu_n = 10 ** (5.2105 + 3.6902 * n)

    f1a = om_m ** (-0.0732)
    f2a = om_m ** (-0.1423)
    f3a = om_m**0.0725
    f1b = om_m ** (-0.0307)
    f2b = om_m ** (-0.0585)
    f3b = om_m ** (0.0743)

    if prescription == "takahashi2012":
        f1 = f1b
        f2 = f2b
        f3 = f3b
    else:
        f1 = frac * f1b + (1 - frac) * f1a
        f2 = frac * f2b + (1 - frac) * f2a
        f3 = frac * f3b + (1 - frac) * f3a

    f = lambda x: x / 4.0 + x**2 / 8.0

    d2l = k**3 * Pk_lin / (2.0 * jnp.pi**2)

    y = k / k_sigma

    # Eq C2; the quasi-linear term uses the Bird et al. (2012) rescaled
    # linear spectrum, which equals d2l when f_nu = 0.
    if prescription == "takahashi2012":
        kh2 = (k / h) ** 2
        d2l_aa = d2l * (1.0 + f_nu * 47.48 * kh2 / (1.0 + 1.5 * kh2))
        halo_nu = 1.0 + 0.977 * f_nu
    else:
        d2l_aa = d2l
        halo_nu = 1.0
    d2q = d2l * ((1.0 + d2l_aa) ** beta_n / (1 + alpha_n * d2l_aa)) * jnp.exp(-f(y))
    d2hprime = (
        a_n * y ** (3 * f1) / (1.0 + b_n * y**f2 + (c_n * f3 * y) ** (3.0 - gamma_n))
    )
    d2h = d2hprime / (1.0 + mu_n / y + nu_n / y**2) * halo_nu
    # Eq. C1
    d2nl = d2q + d2h
    pk_nl = 2.0 * jnp.pi**2 / k**3 * d2nl

    return pk_nl
