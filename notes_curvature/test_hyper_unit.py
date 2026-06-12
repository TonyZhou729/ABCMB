"""
Unit test for the curved-geometry helpers and the hyperspherical Bessel
recurrence used by spectrum._Cl_all_ells_curved.

CPU-only, no GPU needed; runtime ~1-2 min (mpmath reference values).

Checks:
 1. ABCMBTools._curv_f / _curv_g / _curv_g_diff vs 50-digit mpmath, across the
    series/branch switch.
 2. The production recurrence formulas (replicated verbatim) vs
    scipy.special.spherical_jn at K=0.
 3. Same vs a 60-digit mpmath forward recurrence for open and closed K,
    in the region above the evanescent mask (q*S_K(chi) >= flat table edge).
 4. dPhi/d2Phi emission formulas vs mpmath finite differences.
"""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + "/..")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.special import spherical_jn
import mpmath as mp

from abcmb import ABCMBTools as tools
import abcmb.spectrum as spectrum


def xmin_closed_form(ls):
    """Production evanescent threshold (CLASS get_xmin_from_approx), in the
    flat-argument variable; mirrors SpectrumSolver.__init__."""
    lph = np.asarray(ls, dtype=np.float64) + 0.5
    lhs = np.log(2.e-10*lph)/lph
    alpha = -2.*lhs/5.*(1. + 2.*np.cosh(np.arccosh(1. + 375./(16.*lhs*lhs))/3.))
    return lph/np.cosh(alpha)

mp.mp.dps = 60
FAILED = []


def check(name, val, tol):
    status = "PASS" if val <= tol else "FAIL"
    if val > tol:
        FAILED.append(name)
    print(f"  [{status}] {name}: {val:.3e} (tol {tol:.0e})")


### 1. helpers vs mpmath ###
def f_ref(u):
    u = mp.mpf(float(u))
    if u == 0:
        return mp.mpf(1)
    s = mp.sqrt(abs(u))
    return mp.sin(s)/s if u > 0 else mp.sinh(s)/s


def g_ref(u):
    u = mp.mpf(float(u))
    if u == 0:
        return mp.mpf(1)
    s = mp.sqrt(abs(u))
    return s*mp.cos(s)/mp.sin(s) if u > 0 else s*mp.cosh(s)/mp.sinh(s)


us = np.concatenate([
    np.linspace(-2., 2., 161),
    np.geomspace(1e-14, 1.9, 40), -np.geomspace(1e-14, 1.9, 40),
    np.array([0., 0.0099, 0.0101, -0.0099, -0.0101]),  # series boundary
])
print("1. Geometry helpers vs mpmath (dps=60):")
ef = max(abs(float(tools._curv_f(jnp.array(u))) - float(f_ref(u))) for u in us)
eg = max(abs(float(tools._curv_g(jnp.array(u))) - float(g_ref(u))) for u in us)
check("_curv_f abs err", ef, 5e-15)
check("_curv_g abs err", eg, 5e-15)

# g_diff in the cancellation regime: a, b both small
rng = np.random.default_rng(0)
egd = 0.
for _ in range(200):
    a = float(rng.uniform(-1, 1) * 10.**rng.uniform(-14, -2.1))
    b = float(10.**rng.uniform(-14, -2.1))
    ref = float(g_ref(a) - g_ref(b))
    got = float(tools._curv_g_diff(jnp.array(a), jnp.array(b)))
    scale = max(abs(ref), 1e-300)
    egd = max(egd, abs(got - ref)/scale)
check("_curv_g_diff rel err (cancellation regime)", egd, 1e-12)


### Production recurrence, replicated verbatim from _Cl_all_ells_curved ###
def phi_production(K, q, chi, lmax):
    """Phi_l for l = 0..lmax at scalar (q, chi); mirrors the production code."""
    K, q, chi = jnp.float64(K), jnp.float64(q), jnp.float64(chi)
    q2 = q**2          # production uses q2 = k^2 + K with k the grid label
    k = jnp.sqrt(q2 - K)
    uK = K*chi**2
    qchi = q*chi
    Phi0 = jnp.sinc(qchi/jnp.pi) / tools._curv_f(uK)
    Phi1 = Phi0 * tools._curv_g_diff(uK, qchi**2) / (chi*k)
    cotK = tools.cot_K(chi, K)
    out = [Phi0, Phi1]
    Phi_lm1, Phi_l = Phi0, Phi1
    for l in range(1, lmax):
        lf = jnp.float64(l)
        sld = jnp.sqrt(jnp.clip(q2 - K*lf**2, 1.e-30, None))
        slp_arg = q2 - K*(lf+1.)**2
        slpd = jnp.sqrt(jnp.clip(slp_arg, 1.e-30, None))
        Phi_next = jnp.where(slp_arg > 0.,
                             jnp.clip(((2.*lf+1.)*cotK*Phi_l - sld*Phi_lm1)/slpd, -1.e10, 1.e10),
                             0.)
        out.append(Phi_next)
        Phi_lm1, Phi_l = Phi_l, Phi_next
    return np.array([float(x) for x in out])


def dphi_production(K, q, chi, Phi_lm1, Phi_l, l):
    q2 = q**2
    sld = np.sqrt(max(q2 - K*l**2, 1e-30))
    cotK = float(tools.cot_K(jnp.float64(chi), jnp.float64(K)))
    dPhi = sld*Phi_lm1 - (l+1.)*cotK*Phi_l
    sinK = float(tools.sin_K(jnp.float64(chi), jnp.float64(K)))
    d2Phi = -2.*cotK*dPhi + (l*(l+1.)/sinK**2 - q2 + K)*Phi_l
    return dPhi, d2Phi


### mpmath reference ###
def sinK_ref(chi, K):
    chi, K = mp.mpf(chi), mp.mpf(K)
    if K == 0:
        return chi
    s = mp.sqrt(abs(K))
    return mp.sin(s*chi)/s if K > 0 else mp.sinh(s*chi)/s


def cotK_ref(chi, K):
    chi, K = mp.mpf(chi), mp.mpf(K)
    if K == 0:
        return 1/chi
    s = mp.sqrt(abs(K))
    return s*mp.cos(s*chi)/mp.sin(s*chi) if K > 0 else s*mp.cosh(s*chi)/mp.sinh(s*chi)


def phi_ref(K, q, chi, lmax):
    """60-digit forward recurrence with exact seeds (stable at this precision
    for the sampled depths)."""
    K, q, chi = mp.mpf(K), mp.mpf(q), mp.mpf(chi)
    cot = cotK_ref(chi, K)
    Phi0 = mp.sin(q*chi)/(q*sinK_ref(chi, K))
    Phi1 = Phi0*(cot - q*mp.cos(q*chi)/mp.sin(q*chi))/mp.sqrt(q**2 - K)
    out = [Phi0, Phi1]
    for l in range(1, lmax):
        arg = q**2 - K*(l+1)**2
        if arg <= 0:
            out.append(mp.mpf(0))
            out.extend([mp.mpf(0)]*(lmax-l-1))
            break
        nxt = ((2*l+1)*cot*out[-1] - mp.sqrt(q**2 - K*l**2)*out[-2])/mp.sqrt(arg)
        out.append(nxt)
    return out


### 2. flat limit vs scipy ###
# Two tiers: in the oscillatory region (x_eff above the turning point) the
# forward recurrence is stable -> near machine precision. In the transition
# region between the evanescent mask and the turning point, roundoff is
# amplified by the growing solution (design budget: ~1e-16 * e^S with
# |Phi| ~ 1e-10..1 above the mask -> up to ~1e-6 rel-to-peak; enters the Cl
# integral at the same negligible level).
print("2. Recurrence vs scipy spherical_jn at K=0:")
err_osc = err_tra = 0.
for q, chi, lmax in [(0.01, 5000., 80), (0.1, 13000., 1600), (0.25, 13800., 2999)]:
    got = phi_production(0., q, chi, lmax)
    x = q*chi
    ls = np.arange(lmax+1)
    ref = spherical_jn(ls, x)
    xmin = xmin_closed_form(ls)
    scale = np.abs(ref).max()
    relerr = np.abs(got - ref)/scale
    osc = x >= 1.02*np.sqrt(ls*(ls+1.)) + 2.
    tra = (x >= xmin) & ~osc
    if osc.any(): err_osc = max(err_osc, relerr[osc].max())
    if tra.any(): err_tra = max(err_tra, relerr[tra].max())
check("flat vs scipy, oscillatory region", err_osc, 1e-10)
check("flat vs scipy, transition region (above mask)", err_tra, 1e-5)

### 3. curved vs mpmath ###
print("3. Recurrence vs mpmath (open and closed):")
H100 = 3.33564095e-4  # 1/Mpc
for name, omega_k in [("open Ok=+0.05", 0.05*0.6762**2), ("closed Ok=-0.05", -0.05*0.6762**2)]:
    K = -omega_k*H100**2
    err_osc = err_tra = 0.
    for q, chi, lmax in [(3.2e-4, 11000., 60), (2.e-3, 13000., 400), (2.e-2, 9000., 250)]:
        if K > 0 and q**2 - K*lmax**2 <= 0:
            lmax = int(q/np.sqrt(K)) - 1
        got = phi_production(K, q, chi, lmax)
        ref = phi_ref(K, q, chi, lmax)
        x_eff = q*float(sinK_ref(chi, K))
        ls = np.arange(lmax+1)
        xmin = xmin_closed_form(ls)
        reff = np.array([float(r) for r in ref])
        scale = np.abs(reff).max()
        relerr = np.abs(got - reff)/scale
        osc = x_eff >= 1.02*np.sqrt(ls*(ls+1.)) + 2.
        tra = (x_eff >= xmin) & ~osc
        if osc.any(): err_osc = max(err_osc, relerr[osc].max())
        if tra.any(): err_tra = max(err_tra, relerr[tra].max())
    check(f"{name}, oscillatory region", err_osc, 1e-10)
    check(f"{name}, transition region (above mask)", err_tra, 1e-5)

### 4. dPhi / d2Phi emission formulas vs mpmath finite differences ###
print("4. dPhi/d2Phi formulas vs mpmath derivatives:")
err1 = err2 = 0.
for name, omega_k in [("flat", 0.), ("open", 0.05*0.6762**2), ("closed", -0.05*0.6762**2)]:
    K = -omega_k*H100**2
    for q, chi, l in [(2.e-3, 12000., 20), (1.5e-2, 8000., 100)]:
        ref = phi_ref(K, q, chi, l+1)
        Phi_lm1, Phi_l = float(ref[l-1]), float(ref[l])
        dPhi, d2Phi = dphi_production(K, q, chi, Phi_lm1, Phi_l, float(l))
        h = mp.mpf('1e-20')
        f = lambda c: phi_ref(K, q, c, l+1)[l]
        dref = float((f(mp.mpf(chi)+h) - f(mp.mpf(chi)-h))/(2*h))
        d2ref = float((f(mp.mpf(chi)+h) - 2*f(mp.mpf(chi)) + f(mp.mpf(chi)-h))/h**2)
        sc = max(abs(dref), 1e-300)
        err1 = max(err1, abs(dPhi - dref)/sc)
        sc2 = max(abs(d2ref), 1e-300)
        err2 = max(err2, abs(d2Phi - d2ref)/sc2)
check("dPhi rel err", err1, 1e-8)
check("d2Phi rel err", err2, 1e-6)

print()
if FAILED:
    print("FAILED:", FAILED)
    sys.exit(1)
print("ALL PASS")
