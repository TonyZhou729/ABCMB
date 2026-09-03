"""
Script for helper numerical tools
"""
import jax.numpy as jnp
from jax import config, lax, vmap
from jax.scipy.special import factorial

config.update("jax_enable_x64", True)

### BEGINNING OF WIGNER ROTATION FOR LENSING ###

def wigner_d_matrix(mu, ells, m, n):
    """
    Compute Wigner d-matrix elements for rotation.

    Recursively computes reduced Wigner d-matrix elements d^ell_{mn}(beta)
    for CMB lensing calculations using three-term recurrence relation.

    Parameters:
    -----------
    mu : array
        Cosine of rotation angle beta
    ells : array
        Multipole values [m, m+1, m+2, ..., ellmax]
    m : int
        First index (must be positive and >= abs(n))
    n : int
        Second index (must satisfy abs(n) <= m)

    Returns:
    --------
    array
        Wigner d-matrix elements, shape (len(mu), len(ells))
    """
    
    # base case: ell = m
    def base_val(mu):
        beta = jnp.arccos(mu)
        norm = jnp.sqrt((2*m+1)/2) * jnp.sqrt(factorial(2*m)/(factorial(m+n)*factorial(m-n)))
        return norm * jnp.cos(beta/2.)**(m+n)*(-jnp.sin(beta/2.))**(m-n)

    normA = jnp.sqrt((2*ells+3)/(2*ells+1))
    normC = jnp.sqrt((2*ells+3)/(2*ells-1))
    denom = jnp.sqrt((ells+1)**2-m**2) * jnp.sqrt((ells+1)**2-n**2)
    A = jnp.nan_to_num(normA * (ells+1)*(2*ells+1) / denom, 0)
    B = jnp.nan_to_num(-A * m * n / ells / (ells+1), 0)
    C = jnp.nan_to_num(-normC * jnp.sqrt(ells**2-m**2) * jnp.sqrt(ells**2-n**2) / denom * (ells+1)/ells, 0)

    def one_mu(mu):
        d_start = base_val(mu) # Corresponds to ellmin = m
                
        def recursive_dlp1(carry, inputs):
            # For the first iteration, will take d^m_{mn} and d^m_{mn}=0., compute d^{m+1}_{mn}.
            dl, dlm1 = carry 
            a, b, c = inputs

            # Compute dlp1
            dlp1 = a*mu*dl + b*dl + c*dlm1

            # Save dl, then make dl->dlm1, dlp1->dl
            return (dlp1, dl), dl

        # run scan for l = 2..lmax-1
        (_, _), res = lax.scan(recursive_dlp1, (d_start, 0.), (A, B, C))
        return res * jnp.sqrt(2./(2.*ells+1))

    return vmap(one_mu)(mu)

def d00(mu, ells):
    """
    Compute Wigner d-matrix elements d^ell_{00}.

    Parameters:
    -----------
    mu : array
        Cosine of rotation angle
    ells : array
        Multipole values starting from ell=2

    Returns:
    --------
    array
        d^ell_{00} elements for ells >= 2
    """
    # ells go from (2, 3, 4, ..., ellmax)
    ells_patched = jnp.concatenate((jnp.array([0, 1]), ells))
    res = wigner_d_matrix(mu, ells_patched, 0, 0)
    return res[:, 2:] # Return only the ells >= 2

def d1n(mu, ells, n):
    """
    Compute Wigner d-matrix elements d^ell_{1n}.

    Parameters:
    -----------
    mu : array
        Cosine of rotation angle
    ells : array
        Multipole values
    n : int
        Second index (abs(n) <= 1)

    Returns:
    --------
    array
        d^ell_{1n} elements
    """
    # Wigner matrices where m=1, and abs(n)<=m.
    ells_patched = jnp.concatenate((jnp.array([1]), ells))
    res = wigner_d_matrix(mu, ells_patched, 1, n)
    return res[:, 1:]

def d2n(mu, ells, n):
    """
    Compute Wigner d-matrix elements d^ell_{2n}.

    Parameters:
    -----------
    mu : array
        Cosine of rotation angle
    ells : array
        Multipole values
    n : int
        Second index (abs(n) <= 2)

    Returns:
    --------
    array
        d^ell_{2n} elements
    """
    # Wigner matrices where m=2, and abs(n)<=m.
    res = wigner_d_matrix(mu, ells, 2, n)
    return res

def d3n(mu, ells, n):
    """
    Compute Wigner d-matrix elements d^ell_{3n}.

    Parameters:
    -----------
    mu : array
        Cosine of rotation angle
    ells : array
        Multipole values
    n : int
        Second index (abs(n) <= 3)

    Returns:
    --------
    array
        d^ell_{3n} elements, zero-padded for ell < 3
    """
    # Wigner matrices where m=3, and abs(n)<=m.
    ells_sliced = ells[1:] # Compute starting at ell=3
    res = wigner_d_matrix(mu, ells_sliced, 3, n)
    res_patched = jnp.concatenate((jnp.zeros((mu.size, 1)), res), axis=1) # Pad zeros for ell<3.
    return res_patched

def d4n(mu, ells, n):
    """
    Compute Wigner d-matrix elements d^ell_{4n}.

    Parameters:
    -----------
    mu : array
        Cosine of rotation angle
    ells : array
        Multipole values
    n : int
        Second index (abs(n) <= 4)

    Returns:
    --------
    array
        d^ell_{4n} elements, zero-padded for ell < 4
    """
    # Wigner matrices where m=4, and abs(n)<=m.
    ells_sliced = ells[2:] # Compute starting at ell=4
    res = wigner_d_matrix(mu, ells_sliced, 4, n)
    res_patched = jnp.concatenate((jnp.zeros((mu.size, 2)), res), axis=1) # Pad zeros for ell<4.
    return res_patched

### END OF WIGNER ROTATION FOR LENSING ###

### LENSING INTEGRAL QUADRATURE METHODS ###
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

def fast_interp(x, xp_min, xp_max, fp):
    """
    Fast 1D linear interpolation for uniformly-spaced grids.

    Optimized interpolation that avoids searchsorted by exploiting
    uniform grid spacing. Significantly faster than jnp.interp for
    large arrays.

    Parameters:
    -----------
    x : float or array
        Query points for interpolation
    xp_min : float
        Minimum value of interpolation grid
    xp_max : float
        Maximum value of interpolation grid
    fp : array
        Function values on uniform grid

    Returns:
    --------
    float or array
        Interpolated values at query points

    Notes:
    ------
    Credit: JAX issue #16182 (https://github.com/jax-ml/jax/issues/16182)
    Assumes fp is uniformly spaced between xp_min and xp_max.
    """
    # The official jnp.interp is very slow becuase it uses searchsorted.
    # Therefore, we leverage the fact that the fp is linearly increasing, evenly spaced, and has a known range
    # to make this operation much faster.
    eps = 1.e-6
    n = fp.shape[-1]
    i = (x - xp_min) / (xp_max - xp_min) * (n - 1) # fix bug in JAX issue
    i = jnp.clip(i, eps, n - 1.0 - eps)  # Avoid index out of range
    i_lower = jnp.floor(i).astype(jnp.int32)
    i_upper = jnp.minimum(i_lower + 1, n - 1)
    w_upper = i - i_lower
    w_lower = 1.0 - w_upper
    return w_lower * fp[i_lower] + w_upper * fp[i_upper]


def bilinear_interp(x, y, z, xq, yq):
    """
    Bilinear interpolation on 2D regular grid.

    Performs bilinear interpolation to evaluate function at query point
    (xq, yq) given values on a regular 2D grid.

    Parameters:
    -----------
    x : array
        1D array of x-coordinates (must be sorted)
    y : array
        1D array of y-coordinates (must be sorted)
    z : array
        2D array of function values, shape (len(y), len(x))
    xq : float
        Query x-coordinate
    yq : float
        Query y-coordinate

    Returns:
    --------
    float
        Interpolated value at (xq, yq)

    Notes:
    ------
    Uses standard bilinear interpolation formula with four nearest
    grid points.
    """
    # find indices for x and y
    ix = jnp.clip(jnp.searchsorted(x, xq) - 1, 0, x.size - 2)
    iy = jnp.clip(jnp.searchsorted(y, yq) - 1, 0, y.size - 2)

    # grid corner points
    x0, x1 = x[ix], x[ix + 1]
    y0, y1 = y[iy], y[iy + 1]

    # fractional positions
    tx = (xq - x0) / (x1 - x0)
    ty = (yq - y0) / (y1 - y0)

    # get z values
    z00 = z[iy, ix]
    z01 = z[iy, ix + 1]
    z10 = z[iy + 1, ix]
    z11 = z[iy + 1, ix + 1]

    # bilinear interpolation
    return (1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z01 + (1 - tx) * ty * z10 + tx * ty * z11


