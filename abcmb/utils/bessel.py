import numpy as np
import jax
from jax import config
import jax.numpy as jnp
import os
from . import interpolators
file_dir = os.path.dirname(__file__)

config.update("jax_enable_x64", True)

# 2D arrays of tabulated spherical functions over l and x axes.
bessel_l_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/l.txt"), dtype="int")
xphi0_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/xphi0.txt"))
phi0_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/phi0.txt"))
xphi1_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/xphi1.txt"))
phi1_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/phi1.txt"))
xphi2_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/xphi2.txt"))
phi2_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/phi2.txt"))

try:
    gpus = jax.devices('gpu')
    bessel_l_tab = jax.device_put(
        bessel_l_tab, device=gpus[0])
    xphi0_tab = jax.device_put(
        xphi0_tab, device=gpus[0])
    phi0_tab = jax.device_put(
        phi0_tab, device=gpus[0])
    xphi1_tab = jax.device_put(
        xphi1_tab, device=gpus[0])
    phi1_tab = jax.device_put(
        phi1_tab, device=gpus[0])
    xphi2_tab = jax.device_put(
        xphi2_tab, device=gpus[0])
    phi2_tab = jax.device_put(
        phi2_tab, device=gpus[0])
except:
    pass

# More accurate large-x asymptotic expansion of spherical Bessel functions.  Debye's expansion of
# J_{l+1/2}(x), DLMF 10.19.6 https://dlmf.nist.gov/10.19.
def j(l, x):
    """
    Large-x asymptotic spherical Bessel function j_l(x).

    Debye's expansion of J_{l+1/2}(x), used beyond the
    end of the tabulated x range.

    Parameters:
    -----------
    l : int or array
        Multipole
    x : float or array
        Argument, x > l + 1/2

    Returns:
    --------
    float or array
        j_l(x)

    Notes:
    ------
    DLMF 10.19.6 https://dlmf.nist.gov/10.19
    """
    nu = l + 0.5
    s = jnp.sqrt(x**2 - nu**2)                      # nu tan(beta), x = nu sec(beta)
    c = nu / s                                      # cot(beta)
    c2 = c * c
    xi = s - nu*jnp.pi/2 + nu*jnp.arcsin(nu/x) - jnp.pi/4
    cos_t = 1. - c2*(81. + c2*(462. + 385.*c2))/1152./nu**2
    sin_t = c*(3. + 5.*c2)/24./nu - c**3*(30375. + c2*(369603. + c2*(765765. + 425425.*c2)))/414720./nu**3
    return jnp.sqrt(1./(x*s)) * (jnp.cos(xi)*cos_t + jnp.sin(xi)*sin_t)

def j1(x):
    """
    Spherical Bessel function j_1(x) = (sin(x) - x cos(x))/x^2.

    Parameters:
    -----------
    x : float or array
        Argument

    Returns:
    --------
    float or array
        j_1(x)

    Notes:
    ------
    Uses small x series for x^2 < 1e-2.
    """
    # for small x
    series = x/3. * (1 - x**2/10. + x**4/280. - x**6/15120. + x**8/1330560.)
    return jnp.where(x**2 < 1.e-2, series, (jnp.sin(x) - x * jnp.cos(x))/x**2)

def table_indices(l_min, l_max):
    """
    Indices of the tabulated multipoles covering [l_min, l_max].

    Parameters:
    -----------
    l_min : int
        Lowest multipole required
    l_max : int
        Highest multipole required

    Returns:
    --------
    jnp.array
        Consecutive indices into the tables, from the last tabulated l <= l_min
        to the first tabulated l >= l_max
    """
    idx_min = jnp.where(bessel_l_tab<=l_min)[0][-1]
    idx_max = jnp.where(bessel_l_tab>=l_max)[0][0]
    return jnp.arange(idx_min, idx_max+1)

def table_ells(idx):
    """
    Multipoles of the tables at the given indices.

    Parameters:
    -----------
    idx : int or array
        Indices into the tables

    Returns:
    --------
    int or array
        Tabulated multipoles l
    """
    return bessel_l_tab[idx]

def phi_functions(idx):
    """
    Multipole and radial functions for one tabulated multipole.

    Each function is tabulated between its smallest value (~1.e-10) out to the
    fifth local maximum.  Inside his interval fast_interp_cubic is used; below 
    it the function is set to 0, and above it the large-x expansion j is used.

    Parameters:
    -----------
    idx : int
        Index into the tables

    Returns:
    --------
    tuple
        (l, phi0, phi1, phi2), the multipole and the radial functions of x:
        phi0 = j_l, phi1 = j_l', phi2 = (3 j_l'' + j_l)/2

    """
    l = bessel_l_tab[idx]
    x0_min = xphi0_tab[0, idx]
    x0_max = xphi0_tab[-1, idx]
    x1_min = xphi1_tab[0, idx]
    x1_max = xphi1_tab[-1, idx]
    x2_min = xphi2_tab[0, idx]
    x2_max = xphi2_tab[-1, idx]
    col_phi0_l = phi0_tab[:, idx]
    col_phi1_l = phi1_tab[:, idx]
    col_phi2_l = phi2_tab[:, idx]

    def phi0(x):
        x_safe = jnp.where(x >= x0_max, x, x0_max)
        return jnp.where(
            x < x0_min,
            0.,
            jnp.where(
                x >= x0_max,
                j(l, x_safe),
                interpolators.fast_interp_cubic(x, x0_min, x0_max, col_phi0_l)
            )
        )

    def phi1(x):
        x_safe = jnp.where(x >= x1_max, x, x1_max)
        return jnp.where(
            x < x1_min,
            0.,
            jnp.where(
                x >= x1_max,
                l/x_safe*j(l, x_safe) - j(l+1, x_safe),
                interpolators.fast_interp_cubic(x, x1_min, x1_max, col_phi1_l)
            )
        )

    def phi2(x):
        x_safe = jnp.where(x >= x2_max, x, x2_max)
        return jnp.where(
            x < x2_min,
            0.,
            jnp.where(
                x >= x2_max,
                ((3*l*(l-1)-2*x_safe**2)*j(l, x_safe)+6*x_safe*j(l+1, x_safe))/2/x_safe**2,
                interpolators.fast_interp_cubic(x, x2_min, x2_max, col_phi2_l)
            )
        )

    return l, phi0, phi1, phi2
