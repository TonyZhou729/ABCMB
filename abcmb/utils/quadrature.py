import jax
from jax import lax, config
import jax.numpy as jnp
import numpy as np
from scipy.special import erf

config.update("jax_enable_x64", True)

def _pn_and_pnm1_scan(z, n):
    """
    Legendre polynomials P_n(z) and P_{n-1}(z) by upward recurrence.

    Used by gauss_legendre_weights to find the quadrature roots and weights.

    Parameters:
    -----------
    z : float or array
        Points at which to evaluate the polynomials, -1 <= z <= 1
    n : int
        Degree of the highest polynomial, n >= 1

    Returns:
    --------
    (p_n, p_nm1) : (jnp.array, jnp.array)
        P_n(z) and P_{n-1}(z), each with the shape of z
    """
    z = jnp.asarray(z)
    p1 = jnp.ones_like(z)      # P_0
    p2 = jnp.zeros_like(z)     # P_{-1} (dummy)

    def step(carry, j):
        p1, p2 = carry
        # recurrence:
        # new_p1 = P_j, new_p2 = P_{j-1}
        new_p1 = ((2.0*j - 1.0) * z * p1 - (j - 1.0) * p2) / j
        new_p2 = p1
        return (new_p1, new_p2), None

    (p_n, p_nm1), _ = lax.scan(step, (p1, p2), jnp.arange(1, n+1))
    return p_n, p_nm1

def gauss_legendre_weights(n, tol=1.e-16, max_it=50):
    """
    Iteratively finds the roots and weights for Gauss-Legendre quadrature integration
    between -1 and 1, given the number of roots n requested.

    Parameters:
    -----------
    n : int
        Number of roots desired, typically set by lmax of the lensed power spectrum.
    tol : jnp.float64
        Accuracy tolerance on the Newton root finder.
    max_it : int
        Maximum iteration on the Newton root finder.

    Returns:
    --------
    (mu, w) : (jnp.array, jnp.array)
        The roots mu and weights w. 
    """
    dtype=jnp.float64
    m = (n + 1) // 2
    i = jnp.arange(1, m + 1, dtype=dtype)
    z0 = jnp.cos(jnp.array(jnp.pi, dtype=dtype) * (i - 0.25) / (n + 0.5))

    def newton_step(z):
        p_n, p_nm1 = _pn_and_pnm1_scan(z, n)
        pp = n * (z * p_n - p_nm1) / (z*z - 1.0)  # P_n'(z)
        z_new = z - p_n / pp
        return z_new, jnp.max(jnp.abs(z_new - z)), pp

    def cond(state):
        z, err, it = state
        return jnp.logical_and(err > tol, it < max_it)

    def body(state):
        z, err, it = state
        z_new, err_new, _ = newton_step(z)
        return (z_new, err_new, it + 1)

    # init
    z1, err1, _ = newton_step(z0)
    z, err, it = lax.while_loop(cond, body, (z1, err1, jnp.array(1)))

    # final derivative for weights
    p_n, p_nm1 = _pn_and_pnm1_scan(z, n)
    pp = n * (z * p_n - p_nm1) / (z*z - 1.0)
    w_half = 2.0 / ((1.0 - z*z) * pp * pp)

    # match your C layout: mu[i-1] = -z(i), mu[n-i] = z(i)
    mu = jnp.empty((n,), dtype=dtype)
    w  = jnp.empty((n,), dtype=dtype)
    mu = mu.at[:m].set(-z)
    mu = mu.at[n-m:].set(z[::-1])
    w  = w.at[:m].set(w_half)
    w  = w.at[n-m:].set(w_half[::-1])

    return mu, w


def peaked_grid(n, start, center, width, floor, core_below, core_above, taper_above):
    """
    Nodes and weights on [start, 0] with the node density concentrated around
    center, plus a window for moving that concentration (see shift_peaked_grid).

    The node density is floor + exp(-(t - center)^2 / 2 width^2).  Nodes are
    found by inverting the cumulative density with bisection in NumPy, so the
    grid is a fixed function of the arguments.

    Parameters:
    -----------
    n : int
        Number of grid points
    start : float
        First node; the last node is 0
    center : float
        Location of the density peak
    width : float
        Width (standard deviation) of the density peak
    floor : float
        Constant added to the peaked density
    core_below, core_above : float
        The shift window is 1 on [center - core_below, center + core_above]
    taper_above : float
        The shift window tapers (cosine) from 1 to 0 between center - core_below
        and start, and between center + core_above and center + taper_above

    Returns:
    --------
    tuple
        (t0, w0, b, bp, jc):
        t0 : array, nodes, increasing from start to 0
        w0 : array, integration weights of the nodes (trapezoid rule in the
            cumulative-density coordinate)
        b : array, shift window at t0
        bp : array, db/dt at t0
        jc : int, index of a node where b = 1
    """
    r2 = np.sqrt(2.) * width
    cdf = lambda t: floor * (t - start) + width * np.sqrt(np.pi / 2.) * (erf((t - center) / r2) - erf((start - center) / r2))
    M = cdf(0.)
    s = np.linspace(0., 1., n)
    lo, hi = np.full(n, start), np.zeros(n)
    for _ in range(80):  # bisection to machine precision
        mid = 0.5 * (lo + hi)
        below = cdf(mid) < s * M
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    t0 = 0.5 * (lo + hi)
    t0[0], t0[-1] = start, 0.
    w0 = M / (n - 1) / (floor + np.exp(-0.5 * ((t0 - center) / width) ** 2))
    w0[0] *= 0.5
    w0[-1] *= 0.5
    a0, a1 = start, center - core_below
    a2, a3 = center + core_above, center + taper_above
    b, bp = np.zeros(n), np.zeros(n)
    m = (t0 > a0) & (t0 < a1)
    b[m] = 0.5 * (1. - np.cos(np.pi * (t0[m] - a0) / (a1 - a0)))
    bp[m] = 0.5 * np.pi / (a1 - a0) * np.sin(np.pi * (t0[m] - a0) / (a1 - a0))
    b[(t0 >= a1) & (t0 <= a2)] = 1.
    m = (t0 > a2) & (t0 < a3)
    b[m] = 0.5 * (1. + np.cos(np.pi * (t0[m] - a2) / (a3 - a2)))
    bp[m] = -0.5 * np.pi / (a3 - a2) * np.sin(np.pi * (t0[m] - a2) / (a3 - a2))
    return t0, w0, b, bp, int(np.argmin(np.abs(t0 - center)))


def shift_peaked_grid(t0, b, dmu):
    """
    Move the density peak of a peaked_grid by dmu.

    Parameters:
    -----------
    t0 : array
        Nodes from peaked_grid
    b : array
        Shift window from peaked_grid
    dmu : float
        Shift of the peak

    Returns:
    --------
    array
        Shifted nodes t0 + dmu * b(t0)
    """
    return jnp.asarray(t0) + dmu * jnp.asarray(b)


def shifted_grid_weights(t0, w0, bp, jc, t):
    """
    Integration weights for nodes moved by shift_peaked_grid.

    Parameters:
    -----------
    t0, w0, bp, jc :
        Nodes, weights, shift-window derivative and index of a node where the
        window is 1, from peaked_grid
    t : array
        Shifted nodes, t0 + dmu * b(t0)

    Returns:
    --------
    array
        Weight of each node, same shape as t

    Notes:
    ------
    The shift is recovered at a node where the window is 1, dmu = t[jc] - t0[jc].
    The weights are the base weights times the Jacobian of the map,
    w0 * (1 + dmu * b'(t0)), so they remain exact as the nodes move.
    """
    return jnp.asarray(w0) * (1. + (t[jc] - t0[jc]) * jnp.asarray(bp))
