import jax
from jax import lax, config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

def _pn_and_pnm1_scan(z, n):
    """
    Return P_n(z), P_{n-1}(z), Legendre polynomials for vector z using lax.scan.
    Used in function below to find quadrature roots and weights.
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