import numpy as np
import jax.numpy as jnp
import equinox as eqx
import jax
from jax import vmap, jit, config, grad, lax
from diffrax import diffeqsolve, ODETerm, Dopri5, Kvaerno3, Kvaerno5, Tsit5, SaveAt, PIDController, DiscreteTerminatingEvent
from jax.scipy.interpolate import RegularGridInterpolator
from functools import partial
from interpax import CubicSpline
from scipy.special import spherical_jn

from . import ABCMBTools as tools
from . import constants as cnst

import os
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

# large-x asymptotic expansion of spherical bessel functions
Q = lambda l, x : jnp.sqrt(x**2-l**2) - l*jnp.pi/2 + l * jnp.arcsin(l/x)
J = lambda l, x : jnp.sqrt(2/jnp.pi/jnp.sqrt(x**2-l**2)) * jnp.cos(Q(l, x) - jnp.pi/4)
j = lambda l, x : jnp.sqrt(jnp.pi/2/x) * J(l+1/2, x)

def _j1(x):
    """
    Spherical bessel j_1(x) = (sin(x) - x cos(x))/x^2. 

    Notes:
    ------
    Uses small x series for x^2 < 1e-2.
    """
    # for small x
    series = x/3. * (1 - x**2/10. + x**4/280. - x**6/15120. + x**8/1330560.)
    return jnp.where(x**2 < 1.e-2, series, (jnp.sin(x) - x * jnp.cos(x))/x**2)

def phi0(i, x):
    """
    New method for computing phi0 (or just jl)
    We tabulated the bessel function between its smallest value (~1.e-10) out to the fifth local maximum.
    This is a different interval for each l, but we kept identical shape so it can be a large 2D array.
    If the incoming argument is within this interval, we use fast_interp. Otherwise we use the large x expansion above. 
    """
    l = bessel_l_tab[i]
    # use x_safe to avoid double-where gotcha in reverse AD
    x_safe = jnp.where(x >= xphi0_tab[-1, i], x, xphi0_tab[-1, i])
    return jnp.where(
        x < xphi0_tab[0, i],
        0.,
        jnp.where(
            x >= xphi0_tab[-1, i],
            j(l, x_safe),
            tools.fast_interp(x, xphi0_tab[:, i].min(), xphi0_tab[:, i].max(), phi0_tab[:, i])
        )
    )

def phi1(i, x):
    """
    New method for computing phi1, or jl'.
    We tabulated the bessel function between its smallest value (~1.e-10) out to the fifth local maximum.
    This is a different interval for each l, but we kept identical shape so it can be a large 2D array.
    If the incoming argument is within this interval, we use fast_interp. Otherwise we use the large x expansion above. 
    """
    l = bessel_l_tab[i]
    x_safe = jnp.where(x >= xphi1_tab[-1, i], x, xphi1_tab[-1, i])
    return jnp.where(
        x < xphi1_tab[0, i],
        0.,
        jnp.where(
            x >= xphi1_tab[-1, i],
            l/x_safe*j(l, x_safe) - j(l+1, x_safe),
            tools.fast_interp(x, xphi1_tab[:, i].min(), xphi1_tab[:, i].max(), phi1_tab[:, i])
        )
    )

def phi2(i, x):
    """
    New method for computing phi2 = (3 jl'' + jl)/2
    We tabulated the bessel function between its smallest value (~1.e-10) out to the fifth local maximum.
    This is a different interval for each l, but we kept identical shape so it can be a large 2D array.
    If the incoming argument is within this interval, we use fast_interp. Otherwise we use the large x expansion above. 
    """
    l = bessel_l_tab[i]
    x_safe = jnp.where(x >= xphi2_tab[-1, i], x, xphi2_tab[-1, i])
    return jnp.where(
        x < xphi2_tab[0, i],
        0.,
        jnp.where(
            x >= xphi2_tab[-1, i],
            ((3*l*(l-1)-2*x_safe**2)*j(l, x_safe)+6*x_safe*j(l+1, x_safe))/2/x_safe**2,
            tools.fast_interp(x, xphi2_tab[:, i].min(), xphi2_tab[:, i].max(), phi2_tab[:, i])
        )
    )

class SpectrumSolver(eqx.Module):
    """
    CMB angular power spectrum computation.

    Computes temperature and polarization angular power spectra by
    integrating transfer functions over wavenumber and time.

    Attributes:
    -----------
    ells : jnp.array
        Multipole values for output power spectra
    ells_indices : jnp.array
        Indices into bessel_l_tab corresponding to ells
    lensing_ells : jnp.array
        Extended multipole range for lensing calculations
    lensing_ells_indices : jnp.array
        Indices into bessel_l_tab for lensing multipoles
    lensing_mus : jnp.array
        Used for lensing, the Gauss-Legendre quadrature roots for the correlation function -> Cl integral.
    lensing_ws : jnp.array
        Used for lensing, the Gauss-Legendre quadrature weights for the correlation function -> Cl integral.
    lensing : bool
        Whether to include gravitational lensing effects
    use_bessel_tables : bool
        Switch for using tabulated Bessel functions for lensing or compute as part 
        of the lensing calculation.  Tabulated results save ~5% runtime but are 
        only available up to ell_max = 5000.  Defaults to True.
    l_top : int
        Highest multipole of lensing ells (concretized), required for Bessel functions
        computed via recurrence.
    k_axis_transfer : jnp.array
        Wavenumber grid for transfer function integration (units: Mpc^{-1})
    k_axis_Pk_output : jnp.array
        Wavenumber grid for matter power spectrum output (units: Mpc^{-1})
    k_pivot : float
        Pivot scale for primordial power spectrum normalization (units: Mpc^{-1}, default: 0.05)
    scale_sw : float
        Multiplicative factor for Sachs-Wolfe term (default: 1.0)
    scale_isw : float
        Multiplicative factor for integrated Sachs-Wolfe term (default: 1.0)
    scale_dop : float
        Multiplicative factor for Doppler term (default: 1.0)
    scale_pol : float
        Multiplicative factor for polarization term (default: 1.0)
    use_bessel_tables : bool
        If true, uses tabulated bessel function tables, valid up to l=5000. Else bessel functions
        are computed analytically through recurrence relations. 
    delta_l_max : int
        How much to extend the theory Cl l_max if lensing is on, needed for end point convergence in 
        unlensed -> lensed transformation. 
    n_k_cmb : int
        Number of k points over which the CMB source functions should be interpolated over. If lensing
        is on, n_k will be greater than this for accurate lensing potential, but the extra portion is
        irrelevant for just the CMB source part. 

    Methods:
    --------
    primordial_spectrum : Compute primordial power spectrum
    Pk_lin : Compute linear matter power spectrum
    get_Cl : Compute angular power spectra for multiple ℓ
    Cl_one_ell : Compute angular power spectrum for single ℓ
    integrand_T0 : Compute SW+ISW temperature source integrand
    integrand_T1 : Compute ISW temperature source integrand
    integrand_T2 : Compute polarization temperature source integrand
    integrand_E : Compute E-mode polarization source integrand
    """

    ells         : jnp.array
    ells_indices : jnp.array

    lensing_ells : jnp.array
    lensing_ells_indices : jnp.array
    lensing_mus  : jnp.array
    lensing_ws   : jnp.array

    lensing : bool
    use_bessel_tables : bool = eqx.field(static=True)
    l_top : int = eqx.field(static=True)
    n_k_cmb : int = eqx.field(static=True)

    k_axis_transfer  : jnp.array
    k_axis_Pk_output : jnp.array

    k_pivot    : float = 0.05 # In 1/Mpc
    scale_sw  : float = 1.
    scale_isw : float = 1.
    scale_dop : float = 1.
    scale_pol : float = 1.

    def __init__(self,
                 ellmin=2,
                 ellmax=2500,
                 lensing=False,
                 k_axis_transfer=jnp.geomspace(1.e-4, 0.4, 2500),
                 k_axis_Pk_output=jnp.geomspace(1.e-4, 0.1, 100),
                 k_pivot=0.05,
                 scale_sw=1,
                 scale_isw=1,
                 scale_dop=1,
                 scale_pol=1,
                 use_bessel_tables=True,
                 delta_l_max=500,
                 n_k_cmb=-1):
        """
        Initialize CMB spectrum solver.

        Sets up multipole range, lensing configuration, and source term switches
        for computing angular power spectra.

        Parameters:
        -----------
        ellmin : int, optional
            Minimum multipole (default: 2)
        ellmax : int, optional
            Maximum multipole (default: 2500)
        lensing : bool, optional
            Whether to include lensing effects (default: True)
        k_pivot : float, optional
            Pivot scale for primordial spectrum (units: Mpc^{-1}, default: 0.05)
        scale_sw : float, optional
            Switch for Sachs-Wolfe term (default: 1)
        scale_isw : float, optional
            Switch for integrated Sachs-Wolfe term (default: 1)
        scale_dop : float, optional
            Switch for Doppler term (default: 1)
        scale_pol : float, optional
            Switch for polarization term (default: 1)
        use_bessel_tables : bool, optional
            Switch for using tabulated Bessel functions for lensing or compute as part 
            of the lensing calculation.  Tabulated results save ~5% runtime but are 
            only available up to ell_max = 5000.  Defaults to True.
        delta_l_max : int, optional
            Extra multipoles computed above ellmax so the lensed spectra are accurate
            up to ellmax. Same role as CLASS's delta_l_max precision parameter.
            Defaults to 500.
        n_k_cmb : int, optional
            Number of leading nodes of the perturbation k-grid that cover the CMB
            (transfer) range.
        """

        self.lensing = lensing
        self.use_bessel_tables = bool(use_bessel_tables)
        self.n_k_cmb = int(n_k_cmb)

        self.ells = jnp.arange(ellmin, ellmax+1)
        if self.use_bessel_tables:
            ell_idx_min = jnp.where(bessel_l_tab<=ellmin)[0][-1]
            ell_idx_max = jnp.where(bessel_l_tab>=ellmax)[0][0]
            self.ells_indices = jnp.arange(ell_idx_min, ell_idx_max+1)
        else:
            self.ells_indices = jnp.array([0])
        
        if self.lensing:
            lensing_ellmax = ellmax + delta_l_max
            self.lensing_ells = jnp.arange(ellmin, lensing_ellmax+1)
            if self.use_bessel_tables:
                lensing_ell_idx_max = jnp.where(bessel_l_tab>=lensing_ellmax)[0][0]
                self.lensing_ells_indices = jnp.arange(ell_idx_min, lensing_ell_idx_max+1)
            else:
                self.lensing_ells_indices = self.ells_indices # not read if not using bessel tables
            #self.lensing_theta = jnp.linspace(0., jnp.pi/16., lensing_ellmax // 8) # Size recommended by CLASS
            num_mu = lensing_ellmax + 70
            mu, w = tools.gauss_legendre_weights(num_mu)
            self.lensing_mus = jnp.concatenate((mu, jnp.array([1.])))
            self.lensing_ws = jnp.concatenate((w, jnp.array([0.])))
        else:
            self.lensing_ells = self.ells
            self.lensing_ells_indices = self.ells_indices
            #self.lensing_theta = jnp.array([0.]) # Not needed
            self.lensing_mus = jnp.array([0.]) # Not needed
            self.lensing_ws  = jnp.array([0.]) # Not needed

        self.k_axis_transfer = k_axis_transfer
        self.k_axis_Pk_output = k_axis_Pk_output
        self.k_pivot    = k_pivot

        self.scale_sw  = scale_sw
        self.scale_isw = scale_isw
        self.scale_dop = scale_dop
        self.scale_pol = scale_pol

        # need to have this static int stored for Bessel recursion
        self.l_top = int(self.lensing_ells[-1])

    def primordial_spectrum(self, k, params):
        """
        Compute primordial curvature power spectrum.

        Parameters:
        -----------
        k : float or array
            Wavenumber (units: Mpc^{-1})
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        float or array
            Primordial power spectrum P_R(k), units Mpc^3
        """
        return params['A_s']*(k/self.k_pivot)**(params['n_s']-1.) * (2*jnp.pi**2/k**3)

    def Pk_lin(self, k, z, PT, params):
        """
        Compute linear matter power spectrum at wavenumbers k and redshift z.

        Parameters:
        -----------
        k : float or array
            Wavenumber (Mpc^{-1})
        z : float
            Redshift to evaluate.
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        float or array
            Linear matter power spectrum P(k, z), units Mpc^3
        """

        lna = -jnp.log(1.+z)
    
        # vmapped interpolation over Nk (columns of the 2D arrays)
        interp_over_lna = jax.vmap(
            lambda y: jnp.interp(lna, PT.lna, y),
            in_axes=1  # loop over columns
        )

        delta_m_lna = interp_over_lna(PT.delta_m)  # shape (Nk,)

        # Now interpolate over k, in log-log. Log-log matters
        # above the CMB ceiling, where the grid is coarse (~10 nodes/decade).
        delta_m = jnp.exp(
            jnp.interp(jnp.log(k),
                       jnp.log(PT.k),
                       jnp.log(jnp.abs(delta_m_lna)))
        )

        return delta_m**2 * self.primordial_spectrum(k, params)

    def Pk_cb(self, k, z, PT, params):
        """
        Compute linear Baryon+DarkMatter power spectrum at wavenumbers k and redshift z.
        Does not include any other massive species present.

        Parameters:
        -----------
        k : float or array
            Wavenumber (Mpc^{-1})
        z : float
            Redshift to evaluate.
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        float or array
            Linear Baryon+DarkMatter power spectrum P_cb(k, z), units Mpc^3
        """

        lna = -jnp.log(1.+z)
    
        # vmapped interpolation over Nk (columns of the 2D arrays)
        interp_over_lna = jax.vmap(
            lambda y: jnp.interp(lna, PT.lna, y),
            in_axes=1  # loop over columns
        )

        delta_cb_lna = interp_over_lna(PT.delta_cb)

        # now interpolate over k
        delta_cb = jnp.interp(k, PT.k, delta_cb_lna)

        return delta_cb**2 * self.primordial_spectrum(k, params)

    def lensing_power_spectrum(self, k, lna, PT, BG, params):
        """
        Computes the lensing power spectrum at wavenumbers k and redshift z.
        Eq.(3.15) in astro-ph/0601594

        Parameters:
        -----------
        k : float or array
            Wavenumber (Mpc^{-1})
        lna : float
            Scale factor
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : background.Background
            Background cosmology module
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        float or array
            Lensing matter power spectrum P(k, z), dimensionless.
        """
        a = jnp.exp(lna)
        z = 1./a - 1.
        aH = BG.aH(lna, params)

        Omega_m = params["omega_m"]/params["h"]**2
        Omega_L = params["omega_Lambda"]/params["h"]**2
        Omega_r = params["omega_r"]/params["h"]**2

        # Fractional matter density rho_m/rho_tot. This must be the TRUE fraction:
        # the combination Om**2 * aH**4 below reconstructs (4 pi G a^2 rho_m)^2 in
        # the Poisson relation, and that identity only holds if Om = rho_m/rho_tot.
        Om = (Omega_m * (1.+z)**3) / (
            (Omega_m * (1.+z)**3) + Omega_r * (1.+z)**4 + Omega_L
        )

        Pk = self.Pk_lin(k, z, PT, params) # Mpc^3

        return 9./8./jnp.pi**2 * Om**2 * aH**4 * Pk / k

    def lensing_Cl(self, ells, PT, BG, params):
        """
        Angular lensing power spectrum at multipole ell.

        IMPORTANT: Assumes Limber approximation throughout, even at ell=2.

        Eq.(3.14) in astro-ph/0601594, except shifts ell -> ell+1/2 to match CLASS.

        Parameters:
        -----------
        ell : float or array
            Multipole
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : background.Background
            Background cosmology module
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        float or array
            Angular lensing matter power spectrum Cl^phiphi, dimensionless.
        """

        coeff = 8.*jnp.pi**2/(ells+0.5)**3
        chi = lambda lna : BG.tau0 - BG.tau(lna)

        # The previous jnp.nan_to_num(integrand, nan=0.) here masked the
        # forward NaN but left a 0*NaN cotangent in the backward through
        # the where-mask that nan_to_num secretly expands to, which
        # propagated through BG.tau. Fix: substitute lna_safe everywhere,
        # then mask the result to 0 at the boundary.
        lna_axis = jnp.linspace(BG.lna_rec, 0., 500)
        lna_floor = lna_axis[-2]

        def integrand_func(lna):
            lna_safe = jnp.where(lna < 0., lna, lna_floor)
            chi_safe = chi(lna_safe)
            k = (ells+0.5)/chi_safe
            window = (chi(BG.lna_rec) - chi_safe)/chi(BG.lna_rec)/chi_safe
            res = (
                chi_safe / BG.aH(lna_safe, params)
                * window**2
                * self.lensing_power_spectrum(k, lna_safe, PT, BG, params)
            )
            return jnp.where(lna < 0., res, 0.)

        integrand = vmap(integrand_func)(lna_axis)
        return coeff*jnp.trapezoid(integrand, lna_axis, axis=0)

    def lensed_Cls(self, ells, ClTT_unlensed, ClTE_unlensed, ClEE_unlensed, PT, BG, params):
        """
        Compute lensed CMB power spectra.

        Applies gravitational lensing corrections to unlensed temperature
        and polarization power spectra using Wigner rotation matrices.

        Parameters:
        -----------
        ells : array
            Multipole values
        ClTT_unlensed : array
            Unlensed temperature power spectrum
        ClTE_unlensed : array
            Unlensed temperature-E-mode cross spectrum
        ClEE_unlensed : array
            Unlensed E-mode polarization power spectrum
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : background.Background
            Background cosmology module
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        tuple
            (ClTT, ClTE, ClEE) lensed power spectra
        """
        # CLASS samples angle uniformly
        # 500 points is enough for lmax < 4000
        #theta = jnp.linspace(0., jnp.pi/16., 500)

        # Flip mu so that mu is in ascending order, works better for trapz.
        #mu = jnp.flip(jnp.cos(self.lensing_theta))
        mu = self.lensing_mus

        # Compute lensing Cl
        Clpp = self.lensing_Cl(ells, PT, BG, params)

        # Wigner matrices needed in general and for temperature
        # Note that for all wigner matrices, the symmetry relation is dnm = (-1)^(m-n) x dmn
        d00 = tools.d00(mu, ells)
        d11 = tools.d1n(mu, ells, 1)
        d1m1 = tools.d1n(mu, ells, -1)
        d2m2 = tools.d2n(mu, ells, -2)
        dm11 = d1m1

        # Wigner matrices needed for polarization
        d22 = tools.d2n(mu, ells, 2)
        d31 = tools.d3n(mu, ells, 1)
        d40 = tools.d4n(mu, ells, 0)
        d3m3 = tools.d3n(mu, ells, -3)
        d4m4 = tools.d4n(mu, ells, -4)
        d20 = tools.d2n(mu, ells, 0)
        d3m1 = tools.d3n(mu, ells, -1)
        d4m2 = tools.d4n(mu, ells, -2)
        d02 = d20
        dm24 = d4m2

        # Lensing angular correlation function
        Cgl  = 1./4./jnp.pi * jnp.sum(
            (2.*ells+1)*ells*(ells+1)*Clpp*d11, axis=1
        ) # Nmu
        Cgl2 = 1./4./jnp.pi * jnp.sum(
            (2.*ells+1)*ells*(ells+1)*Clpp*dm11, axis=1
        ) # Nmu
        sigma2     = Cgl[-1] - Cgl
        Cgl    = Cgl[:, None]
        Cgl2   = Cgl2[:, None]
        sigma2 = sigma2[:, None]

        llp1   = ells*(ells+1)

        X000       = jnp.exp(-llp1*sigma2/4)
        X000_prime = -llp1/4.*X000
        X220       = 1./4.*jnp.sqrt((ells+2)*(ells-1)*ells*(ells+1))*jnp.exp(-(llp1-2)*sigma2/4.)
        X022       = jnp.exp(-(llp1-4)*sigma2/4)
        X022_prime = -(llp1-4)/4*X022
        X121       = -1./2.*jnp.sqrt((ells+2)*(ells-1))*jnp.exp(-(llp1-8./3.)*sigma2/4.)
        X132       = -1./2.*jnp.sqrt((ells+3)*(ells-2))*jnp.exp(-(llp1-20./3.)*sigma2/4.)
        X242       = 1./4.*jnp.sqrt((ells+4)*(ells+3)*(ells-2)*(ells-3))*jnp.exp(-(llp1-10.)*sigma2/4.)

        # Correlation functions
        ksi = 1./4./jnp.pi * jnp.sum(
            (2.*ells+1)*ClTT_unlensed * (
                X000**2 * d00 \
                + 8./ells/(ells+1)*Cgl2*X000_prime**2*d1m1 \
                + Cgl2**2 * (X000_prime**2*d00 + X220**2*d2m2) \
                #- d00
            ), 
            axis=1
        )

        ksip = 1./4./jnp.pi * jnp.sum(
            (2.*ells+1)*ClEE_unlensed * (
                X022**2 * d22 \
                + 2*Cgl2*X132*X121*d31 \
                + Cgl2**2 * (X022_prime**2*d22 + X242*X220*d40) \
                #- d22
            ), 
            axis=1
        )

        ksim = 1./4./jnp.pi * jnp.sum(
            (2.*ells+1)*ClEE_unlensed * (
                X022**2 * d2m2 \
                + Cgl2*(X121**2*d1m1 + X132**2*d3m3) \
                + 1./2.*Cgl2**2 * (2*X022_prime**2*d2m2 + X220**2*d00 + X242**2*d4m4) \
                #- d2m2
            ), 
            axis=1
        )

        ksix = 1./4./jnp.pi * jnp.sum(
            (2.*ells+1)*ClTE_unlensed * (
                X022*X000*d02 \
                + Cgl2 * 2*X000_prime/jnp.sqrt(llp1) * (X121*d11 + X132*d3m1) \
                + 1./2.*Cgl2**2 * ((2*X022_prime*X000_prime+X220**2)*d20+X220*X242*dm24) \
                #- d02
            ), 
            axis=1
        )
        
        #ClTT = 2.*jnp.pi * jnp.trapezoid(ksi[:, None]*d00, mu, axis=0) + ClTT_unlensed
        #ClTE = 2.*jnp.pi * jnp.trapezoid(ksix[:, None]*d20, mu, axis=0) + ClTE_unlensed
        #ClEE = 1./2. * 2.*jnp.pi * jnp.trapezoid(ksip[:, None]*d22+ksim[:, None]*d2m2, mu, axis=0) + ClEE_unlensed
        w = self.lensing_ws[:, None]
        ClTT = 2*jnp.pi * jnp.sum(ksi[:, None]*d00*w, axis=0)
        ClTE = 2*jnp.pi * jnp.sum(ksix[:, None]*d20*w, axis=0)
        ClEE = 1./2. * 2*jnp.pi * jnp.sum(
            (ksip[:, None]*d22 + ksim[:, None]*d2m2)*w,
            axis=0
        )

        return (ClTT, ClTE, ClEE)

    def get_Cl(self, PT, BG, params):
        """
        Compute angular power spectra for multiple multipoles.

        Parameters:
        -----------
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : background.Background
            Background cosmology module
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        tuple
            (ClTT, ClTE, ClEE) angular power spectra
        """

        if self.use_bessel_tables:
            # _transfer_sources() will be called in Cl_one_ell
            tt_raw, te_raw, ee_raw = vmap(self.Cl_one_ell, in_axes=(0, None, None, None))(self.lensing_ells_indices, PT, BG, params)

            # Cubic spline for smooth Cl over user requested ells
            lensing_ells = bessel_l_tab[self.lensing_ells_indices]
            tt_unlensed = CubicSpline(lensing_ells, tt_raw, check=False)(self.lensing_ells)
            te_unlensed = CubicSpline(lensing_ells, te_raw, check=False)(self.lensing_ells)
            ee_unlensed = CubicSpline(lensing_ells, ee_raw, check=False)(self.lensing_ells)
        
        else:
            sources = self._transfer_sources(PT, BG, params)
            tt_raw, te_raw, ee_raw = self._Cl_all_ells_recurrence(sources, params)
            # recursion will compute all ell, contract to just the lensing ells grid
            offset = (self.l_top - 1) - self.lensing_ells.shape[0]
            # no need to interpolate!
            tt_unlensed = tt_raw[offset:]
            te_unlensed = te_raw[offset:]
            ee_unlensed = ee_raw[offset:]

        def get_lensed_Cls():
            tt_lensed, te_lensed, ee_lensed = self.lensed_Cls(self.lensing_ells, tt_unlensed, te_unlensed, ee_unlensed, PT, BG, params)
            return (tt_lensed[self.ells-2], te_lensed[self.ells-2], ee_lensed[self.ells-2])

        def get_unlensed_Cls():
            return (tt_unlensed[self.ells-2], te_unlensed[self.ells-2], ee_unlensed[self.ells-2])

        return lax.cond(
            self.lensing,
            get_lensed_Cls,
            get_unlensed_Cls
        )

    def Cl_one_ell(self, idx, PT, BG, params):
        """
        Computes angular power spectrum for single multipole.

        Integrates transfer functions over wavenumber.

        Parameters:
        -----------
        idx : int
            Index into bessel_l_tab for multipole ℓ
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : background.Background
            Background cosmology module
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        tuple
            (C_ℓ^TT, C_ℓ^TE, C_ℓ^EE) angular power spectra
        """
        l = bessel_l_tab[idx]
        k_axis = self.k_axis_transfer
        (sourceT0, sourceT1, sourceT2, sourceE), aH_1d, tau, weights, tau0 = self._transfer_sources(PT, BG, params)

        # Here we perform the time integral to get transfer functions from source functions.
        # previously, this block explicitly built a 2D (Nlna, Nk) tensor for each ell and summed it down to (Nk).
        # This newer version refactors into four accumulators of shape (Nk).  For each lna, we compute all four
        # (Nk), multiply by a trapezoid weight, and then add to the accumulator.  The result is identical but
        # avoids having to construct a full 2D tensor for each ell, instead just constructing the 1D (Nk) tensor
        # and accumulating down ell.  Clever "traingle term" added by hand is now handled by the trapezoid weights.

        # Pre-slice bessel-table columns so the scan body doesn't re-index
        # ..._tab[:, idx] every iteration.
        x0_min = xphi0_tab[0, idx]
        x0_max = xphi0_tab[-1, idx]
        x1_min = xphi1_tab[0, idx]
        x1_max = xphi1_tab[-1, idx]
        x2_min = xphi2_tab[0, idx]
        x2_max = xphi2_tab[-1, idx]
        col_phi0_l = phi0_tab[:, idx]
        col_phi1_l = phi1_tab[:, idx]
        col_phi2_l = phi2_tab[:, idx]
        ell_eps_factor = jnp.sqrt(3./8.*(l+2)*(l+1)*l*(l-1))

        def phi0_local(x):
            x_safe = jnp.where(x >= x0_max, x, x0_max)
            return jnp.where(
                x < x0_min,
                0.,
                jnp.where(
                    x >= x0_max,
                    j(l, x_safe),
                    tools.fast_interp(x, x0_min, x0_max, col_phi0_l)
                )
            )

        def phi1_local(x):
            x_safe = jnp.where(x >= x1_max, x, x1_max)
            return jnp.where(
                x < x1_min,
                0.,
                jnp.where(
                    x >= x1_max,
                    l/x_safe*j(l, x_safe) - j(l+1, x_safe),
                    tools.fast_interp(x, x1_min, x1_max, col_phi1_l)
                )
            )

        def phi2_local(x):
            x_safe = jnp.where(x >= x2_max, x, x2_max)
            return jnp.where(
                x < x2_min,
                0.,
                jnp.where(
                    x >= x2_max,
                    ((3*l*(l-1)-2*x_safe**2)*j(l, x_safe)+6*x_safe*j(l+1, x_safe))/2/x_safe**2,
                    tools.fast_interp(x, x2_min, x2_max, col_phi2_l)
                )
            )

        zero_k = jnp.zeros(k_axis.shape, dtype=sourceT0.dtype)

        def scan_step(carry, xs_l):
            acc_T0, acc_T1, acc_T2, acc_E = carry
            sT0_l, sT1_l, sT2_l, sE_l, aH_l, tau_l, w_l = xs_l
            chi_l = (tau0 - tau_l) * k_axis
            phi0_l = phi0_local(chi_l)
            phi1_l = phi1_local(chi_l)
            phi2_l = phi2_local(chi_l)
            eps_l  = phi0_l / chi_l**2 * ell_eps_factor
            inv_aH = 1.0 / aH_l
            acc_T0 = acc_T0 + w_l * sT0_l * inv_aH * phi0_l
            acc_T1 = acc_T1 + w_l * sT1_l * inv_aH * phi1_l
            acc_T2 = acc_T2 + w_l * sT2_l * inv_aH * phi2_l
            acc_E  = acc_E  + w_l * sE_l  * inv_aH * eps_l
            return (acc_T0, acc_T1, acc_T2, acc_E), None

        init = (zero_k, zero_k, zero_k, zero_k)
        xs = (sourceT0, sourceT1, sourceT2, sourceE, aH_1d, tau, weights)
        # jax.checkpoint on the scan body: during reverse AD, body intermediates
        # are not saved — the body is re-executed on the backward pass. Kills
        # the ~21 GiB (Nell, Nlna, Nk) integrand rematerialisation; adds ~2× on
        # this scan's compute, a small fraction of SS wall time.
        (transferT0, transferT1, transferT2, transferE), _ = lax.scan(
            jax.checkpoint(scan_step), init, xs
        )

        transferT = transferT0 + transferT1 + transferT2
        ### END OF TRANSFER FUNCTION ###

        # Now we integrate the transfer functions along the line of sight, and return. 
        integrandTT = 4.*jnp.pi * params['A_s'] * (k_axis/self.k_pivot)**(params['n_s']-1.) * transferT**2 / k_axis
        integrandTE = 4.*jnp.pi * params['A_s'] * (k_axis/self.k_pivot)**(params['n_s']-1.) * transferT*transferE / k_axis
        integrandEE = 4.*jnp.pi * params['A_s'] * (k_axis/self.k_pivot)**(params['n_s']-1.) * transferE**2 / k_axis
        
        return (
            jnp.trapezoid(integrandTT, k_axis),
            jnp.trapezoid(integrandTE, k_axis),
            jnp.trapezoid(integrandEE, k_axis)
        )
    
    # CG: reorganizing this section into its own method
    def _transfer_sources(self, PT, BG, params):
            """
            Assemble the line-of-sight source functions on (lna, k_transfer) grid.

            Parameters:
            -----------
            PT : perturbations.PerturbationTable
                Perturbation evolution table
            BG : background.Background
                Background cosmology module
            params : dict
                Dictionary of input and derived parameters

            Returns:
            -------
            tuple
                (sourceT0, sourceT1, sourceT2, sourceE) of shape (Nlna, Nk), plus
                aH_1d, tau, weights (Nlna,) and tau0.
            """
            k_axis = self.k_axis_transfer 
            lna_axis = PT.lna[:-1]
            delta_lna = PT.lna[-1] - PT.lna[-2]

            # Background quantities, all Nlna 1D vectors
            tau0 = BG.tau0
            tau = BG.tau(lna_axis)
            g   = vmap(BG.visibility,in_axes=[0,None])(lna_axis, params)
            g_prime = vmap(grad(BG.visibility,argnums=0),in_axes=[0,None])(lna_axis, params) # Derivative of g w.r.t. lna
            aH  = BG.aH(lna_axis, params)
            expmkappa = vmap(BG.expmkappa)(lna_axis)
            aH_dot = BG.aH_prime(lna_axis, params) * aH # Derivative of aH w.r.t. conformal time tau.

            # Keep a 1D alias of aH for the rolling-accumulator scan below.
            aH_1d = aH

            g         = g[:, None]
            g_prime   = g_prime[:, None]
            aH        = aH[:, None]
            expmkappa = expmkappa[:, None]
            aH_dot    = aH_dot[:, None]

            # Perturbations, all (Nlna, Nk) 2D vectors
            # Cubic Spline is necessary here for accuracy. 
            # Restrict the spline to the CMB portion of the perturbation grid.
            # Above it the grid switches to coarse logarithmic steps for the
            # lensing potential.
            n_cmb = self.n_k_cmb if self.n_k_cmb > 0 else PT.k.shape[0]
            log_k_cmb = jnp.log10(PT.k[:n_cmb])
            interp_column = lambda col : CubicSpline(log_k_cmb, col[:n_cmb], check=False)(jnp.log10(k_axis))

            # Found that this is much much faster than RegularGridInterpolator
            photon_sp = PT.species_perturbations["Photon"]
            baryon_sp = PT.species_perturbations["Baryon"]
            delta_g       = vmap(interp_column, in_axes=0, out_axes=0)(photon_sp["delta"][:-1, :])
            theta_b       = vmap(interp_column, in_axes=0, out_axes=0)(baryon_sp["theta"][:-1, :])
            theta_b_prime = vmap(interp_column, in_axes=0, out_axes=0)(PT.theta_b_prime[:-1, :])
            sigma_g       = vmap(interp_column, in_axes=0, out_axes=0)(photon_sp["sigma"][:-1, :])
            Gg0           = vmap(interp_column, in_axes=0, out_axes=0)(photon_sp["G0"][:-1, :])
            Gg2           = vmap(interp_column, in_axes=0, out_axes=0)(photon_sp["G2"][:-1, :])
            eta           = vmap(interp_column, in_axes=0, out_axes=0)(PT.metric_eta[:-1, :])
            eta_prime     = vmap(interp_column, in_axes=0, out_axes=0)(PT.metric_eta_prime[:-1, :])
            alpha         = vmap(interp_column, in_axes=0, out_axes=0)(PT.metric_alpha[:-1, :])
            alpha_prime   = vmap(interp_column, in_axes=0, out_axes=0)(PT.metric_alpha_prime[:-1, :])

            # Source terms
            sourceT0 = self.scale_sw * g * (delta_g/4. + aH*alpha_prime) \
                    + self.scale_isw * (
                        g * (eta - aH*alpha_prime - 2.*aH*alpha) \
                        + 2.*expmkappa * (aH*eta_prime - aH_dot*alpha - aH**2*alpha_prime)
                    ) \
                    + self.scale_dop * (
                        aH * (g*((theta_b_prime / k_axis**2) + alpha_prime) \
                        + g_prime*((theta_b / k_axis**2) + alpha))
                    )

            sourceT1 = self.scale_isw * expmkappa * \
                    ((aH*alpha_prime + 2.*aH*alpha - eta) * k_axis)

            sourceT2 = self.scale_pol * g * (2*sigma_g + Gg0 + Gg2) / 8.

            sourceE  = jnp.sqrt(6) * g * (2*sigma_g + Gg0 + Gg2) / 8.

            Nlna = lna_axis.shape[0]
            weights = jnp.full((Nlna,), delta_lna, dtype=sourceT0.dtype)
            weights = weights.at[0].set(0.5 * delta_lna)

            return (sourceT0, sourceT1, sourceT2, sourceE), aH_1d, tau, weights, tau0

    def _Cl_all_ells_recurrence(self, sources, params):
        """
        Compute Cl at each integer ell using the recursion relation for spherical
        Bessel functions

            j_{l+1}(x) = (2l+1)/x j_l(x) - j_{l-1}(x),   x = k chi

        as opposed to the tabulated Bessel functions.

        Parameters:
        -----------
        sources : tuple
            Output of _transfer_sources (source functions for transfer functions)
        params : dict
            Dictionary of the input and derived parameters.

        Returns:
        --------
        tuple
            (ClTT, ClTE, CLEE) for ell between ell=2 (regardless of ell_min) and l_top

        
        Notes:
        ------
        Stable for x > l.  Error is clamped for x < l, and a zero-mask is applied
        below x_min(l), where |j_l| ~ 1e-10.
        """
        (sourceT0, sourceT1, sourceT2, sourceE), aH_1d, tau, weights, tau0 = sources
        k_axis = self.k_axis_transfer           # (Nk,)

        # same operations as scan_step above
        chi = (tau0 - tau)[:, None]             # (Nlna, 1)
        x = k_axis * chi                        # (Nlna, Nk)
        inv_x = 1./x
        inv_x2 = 1./x**2

        # set up recursion
        Phi0 = jnp.sinc(x/jnp.pi)
        Phi1 = _j1(x)
        Phi2 = 3. * inv_x * Phi1 - Phi0

        # e.g. w_l * sT0_l * inv_aH in Cl_one_ell is row l of SW0 here
        wa = (weights/aH_1d)[:,None]
        SW0 = sourceT0 * wa
        SW1 = sourceT1 * wa
        SW2 = sourceT2 * wa
        SWE = sourceE * wa

        # k integral is the same trapezoid in Cl_one_ell; see e.g. integrandTT
        dk = jnp.diff(k_axis)
        wk = jnp.concatenate((dk[:1]/2., (dk[1:] + dk[:-1])/2., dk[-1:]/2.))
        wk_prim = wk * 4.*jnp.pi * params['A_s'] * (k_axis/self.k_pivot)**(params['n_s']-1.) / k_axis

        # with weights precomputed, sum over lna then sum over k
        def step(carry, xs_l):
            Phi_lm1, Phi_l = carry
            lf, xmin_l = xs_l # lf is current ell as a float

            # phi0 = j_l, phi1 = j_l', phi2 = (3 j_l'' + j_l)/2, 
            # eps = sqrt(3/8 (l+2)!/(l-2)!) j_l/x^2, as in Cl_one_ell.
            # j_l and its derivatives are now computed sequentially in this loop.
            j_prime = Phi_lm1 - (lf+1.)*inv_x*Phi_l
            j_2prime = -2*inv_x*j_prime + (lf*(lf+1.)*inv_x2 - 1.)*Phi_l
            mask = x >= xmin_l

            # get masked bessel arrays
            r0 = jnp.where(mask, Phi_l, 0.)
            r1 = jnp.where(mask, j_prime, 0.)
            r2 = jnp.where(mask, (3.*j_2prime + Phi_l)/2., 0.)
            eps_factor = jnp.sqrt(3./8.*(lf+2.)*(lf+1.)*lf*(lf-1.))
            rE = eps_factor * jnp.where(mask, Phi_l * inv_x2, 0.)

            transferT = jnp.sum(SW0*r0 + SW1*r1 + SW2*r2, axis=0)  # sum lna. (Nk,)
            transferE = jnp.sum(SWE*rE, axis=0)

            clTT = jnp.sum(wk_prim*transferT**2) # sum in k
            clTE = jnp.sum(wk_prim*transferT*transferE)
            clEE = jnp.sum(wk_prim*transferE**2)
    
            # clip masked growth from instability of recursion
            Phi_next = jnp.where(mask, (2.*lf + 1.)*inv_x*Phi_l - Phi_lm1, 0.)
            return (Phi_l, Phi_next), jnp.stack((clTT, clTE, clEE))

        # Set up and run the loop.  Memory-aware to avoid spoiling reverse AD
        CHUNK = 64
        n = self.l_top - 1
        npad = (-n) % CHUNK
        # This is hyperspherical_get_xmin_from_approx in CLASS for K = 0, find xmin_l for the loops above
        lf_all = np.arange(2, 2 + n + npad, dtype=np.float64)
        lph = lf_all + 0.5
        lhs = np.log(2.e-10*lph)/lph
        alpha = -2.*lhs/5.*(1. + 2.*np.cosh(np.arccosh(1. + 375./(16.*lhs*lhs))/3.))
        xmin_all = lph/np.cosh(alpha)
        xs = (jnp.asarray(lf_all).reshape(-1, CHUNK), jnp.asarray(xmin_all).reshape(-1, CHUNK))

        # run the loop
        def chunk_body(carry, xs_chunk):
            return lax.scan(step, carry, xs_chunk)

        # Chunked scan over l and reshaping
        _, outs = lax.scan(jax.checkpoint(chunk_body), (Phi1, Phi2), xs)
        cls = outs.reshape(-1, 3)[:n]
        return cls[:, 0], cls[:, 1], cls[:, 2]