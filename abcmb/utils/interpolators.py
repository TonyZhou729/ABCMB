import jax
from jax import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

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

def fast_interp_cubic(x, xp_min, xp_max, fp):
    """
    Cubic version of fast_interp to avoid gradient discontinuities at 
    grid points.

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

    """
    eps = 1.e-6
    n = fp.shape[-1]
    i = (x - xp_min) / (xp_max - xp_min) * (n - 1) # fix bug in JAX issue
    i = jnp.clip(i, eps, n - 1.0 - eps)  # Avoid index out of range
    i1 = jnp.floor(i).astype(jnp.int32)
    t = i - i1
    p0 = fp[jnp.maximum(i1-1,0)]
    p1 = fp[i1]
    p2 = fp[jnp.minimum(i1+1, n-1)]
    p3 = fp[jnp.minimum(i1+2, n-1)]
    return p1 + 0.5 * t * (p2 - p0 + t * (2.*p0 - 5.*p1 + 4.*p2 - p3 + t * (3.*(p1 - p2) + p3 - p0)))


def loglog_interp(x, xp, fp):
    """
    1D interpolation that is linear in (ln x, ln f).

    Parameters:
    -----------
    x : float or array
        Query points, x > 0
    xp : array
        Increasing grid, xp > 0
    fp : array
        Function values on the grid, fp > 0

    Returns:
    --------
    float or array
        Interpolated values at query points
    """
    return jnp.exp(jnp.interp(jnp.log(x), jnp.log(xp), jnp.log(fp)))


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