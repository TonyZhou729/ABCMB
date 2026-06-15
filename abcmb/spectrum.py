import numpy as np
import jax.numpy as jnp
import equinox as eqx
import jax
from jax import vmap, jit, config, grad, lax
from diffrax import diffeqsolve, ODETerm, Dopri5, Kvaerno3, Kvaerno5, Tsit5, SaveAt, PIDController, DiscreteTerminatingEvent
from jax.scipy.interpolate import RegularGridInterpolator
from functools import partial
from interpax import CubicSpline

from . import ABCMBTools as tools
from . import constants as cnst

import os
file_dir = os.path.dirname(__file__)

config.update("jax_enable_x64", True)

# Spherical-Bessel radial functions are generated on the fly by the
# hyperspherical recurrence in SpectrumSolver._Cl_all_ells_curved, which is
# smooth through K=0 (there it produces exactly j_l(k chi)). There are no
# tabulated Bessel tables, no large-x asymptotic, and no sparse-l spline.

class SpectrumSolver(eqx.Module):
    """
    CMB angular power spectrum computation.

    Computes temperature and polarization angular power spectra by
    integrating transfer functions over wavenumber and time.

    Attributes:
    -----------
    ells : jnp.array
        Multipole values for output power spectra
    lensing_ells : jnp.array
        Extended multipole range for lensing calculations
    lensing_mus : jnp.array
        Used for lensing, the Gauss-Legendre quadrature roots for the correlation function -> Cl integral.
    lensing_ws : jnp.array
        Used for lensing, the Gauss-Legendre quadrature weights for the correlation function -> Cl integral.
    lensing : bool
        Whether to include gravitational lensing effects
    curvature : bool
        Whether to build the curved-geometry k-grid (required whenever
        omega_k != 0). The hyperspherical-Bessel line-of-sight recurrence in
        _Cl_all_ells_curved is used for every cosmology (it reduces to j_l at
        K=0), so this flag no longer selects the spectrum path (static).
    curv_ells : jnp.array
        Integer multipoles 2..lensing_ells[-1] emitted by the recurrence
    curv_xmin : jnp.array
        Per-ell evanescent cutoff in the variable q*S_K(chi), from CLASS's
        closed-form estimate (hyperspherical_get_xmin_from_approx)
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

    Methods:
    --------
    primordial_spectrum : Compute primordial power spectrum
    Pk_lin : Compute linear matter power spectrum
    get_Cl : Compute angular power spectra for multiple ℓ
    integrand_T0 : Compute SW+ISW temperature source integrand
    integrand_T1 : Compute ISW temperature source integrand
    integrand_T2 : Compute polarization temperature source integrand
    integrand_E : Compute E-mode polarization source integrand
    """

    ells         : jnp.array

    lensing_ells : jnp.array
    lensing_mus  : jnp.array
    lensing_ws   : jnp.array

    lensing : bool
    curvature : bool = eqx.field(static=True)

    k_axis_transfer  : jnp.array
    k_axis_Pk_output : jnp.array

    curv_ells : jnp.array
    curv_xmin : jnp.array

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
                 curvature=False):
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
        """

        self.lensing = lensing

        # Every integer multipole is emitted directly by the recurrence in
        # _Cl_all_ells_curved (see curv_ells below), so there is no sparse-ell
        # subset to index into and no Bessel table to look ells up in.
        self.ells = jnp.arange(ellmin, ellmax+1)

        if self.lensing:
            lensing_ellmax = ellmax+500
            self.lensing_ells = jnp.arange(ellmin, lensing_ellmax+1)
            #self.lensing_theta = jnp.linspace(0., jnp.pi/16., lensing_ellmax // 8) # Size recommended by CLASS
            num_mu = lensing_ellmax + 70
            mu, w = tools.gauss_legendre_weights(num_mu)
            self.lensing_mus = jnp.concatenate((mu, jnp.array([1.])))
            self.lensing_ws = jnp.concatenate((w, jnp.array([0.])))
        else:
            self.lensing_ells = self.ells
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

        # curv_ells: every integer ell emitted by the recurrence. curv_xmin:
        # per-ell evanescent cutoff in q S_K(chi) (|Phi| < 1e-10), from CLASS's
        # closed form hyperspherical_get_xmin_from_approx.
        self.curvature = bool(curvature)
        l_top = int(self.lensing_ells[-1])
        self.curv_ells = jnp.arange(2, l_top+1)
        lph = np.arange(2, l_top+1, dtype=np.float64) + 0.5
        lhs = np.log(2.e-10*lph)/lph
        alpha = -2.*lhs/5.*(1. + 2.*np.cosh(np.arccosh(1. + 375./(16.*lhs*lhs))/3.))
        self.curv_xmin = jnp.array(lph/np.cosh(alpha))

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

        # now interpolate over k
        delta_m = jnp.interp(k, PT.k, delta_m_lna)

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

        delta_dm_lna = interp_over_lna(PT.species_perturbations["ColdDarkMatter"]["delta"])
        delta_b_lna  = interp_over_lna(PT.species_perturbations["Baryon"]["delta"])

        # now interpolate over k
        delta_dm = jnp.interp(k, PT.k, delta_dm_lna)
        delta_b  = jnp.interp(k, PT.k, delta_b_lna)

        # total matter overdensity
        delta_m = (
            params['omega_b']   * delta_b +
            params['omega_cdm'] * delta_dm
        ) / params['omega_m']

        return delta_m**2 * self.primordial_spectrum(k, params)

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
        Omega_k = params["omega_k"]/params["h"]**2
        Omega_L = params["omega_Lambda"]/params["h"]**2

        # Matter fraction over time after equality. 1 at early times and becomes Om0 today.
        Om = (Omega_m * (1.+z)**3)/ ((Omega_m * (1.+z)**3) + Omega_k * (1.+z)**2 + Omega_L)

        Pk = self.Pk_lin(k, z, PT, params) # Mpc^3

        # Curved Poisson equation: (k^2 - 3K) Psi = -4 pi G a^2 rho delta,
        # so the flat 1/k becomes k^3/(k^2-3K)^2.
        K = params['K']
        return 9./8./jnp.pi**2 * Om**2 * aH**4 * Pk * k**3 / (k**2 - 3.*K)**2

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

        # Curved-sky Limber: C_l = 4 int dchi W^2/S_K(chi)^2 P_Psi,3D(k(chi))
        # with q = (l+1/2)/S_K(chi), k = sqrt(q^2 - K), the curved lensing
        # kernel W = S_K(chi*-chi)/(S_K(chi*) S_K(chi)), and the hyperspherical
        # WKB amplitude correction (1 - K l^2/q^2)^(-1/2) (CLASS
        # transfer_limber). Reduces exactly to the flat
        # 8 pi^2/(l+1/2)^3 int dchi chi W^2 P_Psi form at K = 0.
        K = params['K']
        coeff = 8.*jnp.pi**2
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
            chi_star = chi(BG.lna_rec)
            sK      = tools.sin_K(chi_safe, K)
            sK_star = tools.sin_K(chi_star, K)
            q = (ells+0.5)/sK
            k = jnp.sqrt(jnp.clip(q**2 - K, 1.e-30, None))
            window = tools.sin_K(chi_star - chi_safe, K)/sK_star/sK
            wkb_amp = 1./jnp.sqrt(jnp.clip(1. - K*ells**2/q**2, 1.e-30, None))
            res = (
                1. / BG.aH(lna_safe, params)
                * window**2 / (sK**2 * k**3)
                * wkb_amp
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

    def _transfer_sources(self, PT, BG, params):
        """
        Assemble the line-of-sight source functions on the (lna, k_transfer) grid.

        Curvature enters only via the PerturbationTable metric quantities and the
        s_2 = sqrt(1-3K/k^2) polarization factor (s_2 = 1 in the flat limit).

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
            (sourceT0..E) of shape (Nlna, Nk), plus aH_1d, tau, weights (Nlna,) and tau0.
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

        # Keep a 1D alias of aH for the rolling-accumulator scan downstream.
        aH_1d = aH

        g         = g[:, None]
        g_prime   = g_prime[:, None]
        aH        = aH[:, None]
        expmkappa = expmkappa[:, None]
        aH_dot    = aH_dot[:, None]

        # Perturbations, all (Nlna, Nk) 2D vectors
        # Cubic Spline is necessary here for accuracy.
        interp_column = lambda col : CubicSpline(jnp.log10(PT.k), col, check=False)(jnp.log10(k_axis))

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

        # Curved polarization weight: s_2 dresses sigma_g inside Pi (=1 at K=0).
        s2 = jnp.sqrt(jnp.clip(1. - 3.*params['K']/k_axis**2, 1.e-30, None))

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

        sourceT2 = self.scale_pol * g * (2*s2*sigma_g + Gg0 + Gg2) / 8.

        sourceE  = jnp.sqrt(6) * g * (2*s2*sigma_g + Gg0 + Gg2) / 8.

        # Trapezoid weights over the (uniform) lna grid; the first point gets
        # the half weight, the last grid point (lna = 0, chi = 0) is excluded
        # from lna_axis and its triangle correction is carried by delta_lna.
        Nlna = lna_axis.shape[0]
        weights = jnp.full((Nlna,), delta_lna, dtype=sourceT0.dtype)
        weights = weights.at[0].set(0.5 * delta_lna)

        return (sourceT0, sourceT1, sourceT2, sourceE), aH_1d, tau, weights, tau0

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

        sources = self._transfer_sources(PT, BG, params)

        # Exact hyperspherical-Bessel recurrence: Cl at every integer ell from 2
        # to lensing_ells[-1], for every cosmology (it reduces to the flat j_l
        # at K=0). No sparse-ell spline reconstruction. Static shape arithmetic:
        # curv_ells = arange(2, l_top+1) and lensing_ells = arange(ellmin,
        # l_top+1), so the offset of ellmin into curv_ells is the length
        # difference.
        tt_all, te_all, ee_all = self._Cl_all_ells_curved(sources, params)
        off = self.curv_ells.shape[0] - self.lensing_ells.shape[0]
        tt_unlensed = tt_all[off:]
        te_unlensed = te_all[off:]
        ee_unlensed = ee_all[off:]

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

    def _Cl_all_ells_curved(self, sources, params):
        """
        Cl at every integer ell via the exact hyperspherical-Bessel recurrence
        (Lesgourgues & Tram 1305.3261; Tram 1311.0839)

            sqrt(q^2 - K l^2) Phi_l = (2l-1) cot_K(chi) Phi_{l-1}
                                      - sqrt(q^2 - K (l-1)^2) Phi_{l-2},  q^2 = k^2 + K,

        seeded by Phi_0 = sin(q chi)/(q S_K(chi)), Phi_1 = Phi_0 (cot_K - q cot q chi)/k.
        Smooth through K = 0 (-> j_l(k chi)). One chunked lax.scan (jax.checkpoint
        per chunk) walks l upward, contracting (Phi_{l-1}, Phi_l) against `sources`
        — exact per ell, no spline. Evanescent values (below the turning point
        q S_K(chi) = sqrt(l(l+1))) are clamped and masked to zero at |Phi| ~ 1e-10;
        closed modes terminate at l >= nu = q/sqrt(K), open q^2 <= 0 modes carry
        zero k-weight.

        Parameters:
        -----------
        sources : tuple
            Output of _transfer_sources.
        params : dict
            Dictionary of input and derived parameters.

        Returns:
        --------
        tuple
            (ClTT, ClTE, ClEE) on arange(2, lensing_ells[-1]+1).
        """
        (sourceT0, sourceT1, sourceT2, sourceE), aH_1d, tau, weights, tau0 = sources
        k_axis = self.k_axis_transfer                       # (Nk,)
        K = params['K']

        chi  = (tau0 - tau)[:, None]                        # (Nlna, 1)
        q2 = k_axis**2 + K                                  # (Nk,)
        qmask = (q2 > 0.)
        q  = jnp.sqrt(jnp.clip(q2, 1.e-30, None))
        s2 = jnp.sqrt(jnp.clip(1. - 3.*K/k_axis**2, 1.e-30, None))

        sinK = tools.sin_K(chi, K)                          # (Nlna, 1)
        cotK = tools.cot_K(chi, K)                          # (Nlna, 1)
        uK   = K*chi**2
        qchi = q*chi                                        # (Nlna, Nk)

        # Seeds (sqrt(q^2 - K) = k exactly; _curv_g_diff is cancellation-safe).
        Phi0 = jnp.sinc(qchi/jnp.pi) / tools._curv_f(uK)
        Phi1 = Phi0 * tools._curv_g_diff(uK, qchi**2) / (chi*k_axis)
        s1d  = k_axis
        # Relative threshold separating physical modes from FP noise at the
        # closed-universe termination l+1 = nu (where q^2 - K(l+1)^2 -> 0).
        term_tol = 1.e-6*q2
        s2d_arg = q2 - 4.*K
        s2d  = jnp.sqrt(jnp.clip(s2d_arg, 1.e-30, None))
        Phi2 = jnp.where(s2d_arg > term_tol,
                         jnp.clip((3.*cotK*Phi1 - s1d*Phi0)/s2d, -1.e10, 1.e10),
                         0.)

        # Sources pre-multiplied by trapezoid weights / aH; wk_prim carries the
        # dk/k measure and the primordial power law.
        wa  = (weights/aH_1d)[:, None]
        SW0 = sourceT0*wa
        SW1 = sourceT1*wa
        SW2 = sourceT2*wa
        SWE = sourceE*wa

        dk = jnp.diff(k_axis)
        wk = jnp.concatenate((dk[:1]/2., (dk[1:]+dk[:-1])/2., dk[-1:]/2.))
        wk_prim = wk * 4.*jnp.pi * params['A_s'] * (k_axis/self.k_pivot)**(params['n_s']-1.) / k_axis \
                  * qmask

        # Evanescent cutoff variable q S_K(chi) (= k chi in the flat limit).
        x_eff = q*sinK                                      # (Nlna, Nk)

        def step(carry, xs_l):
            Phi_lm1, Phi_l = carry
            lf, xmin_l = xs_l

            # Radial functions T0=Phi, T1=Phi'/k, T2=(3 Phi''/k^2 + Phi)/(2 s2),
            # E = sqrt(3/8 (l+2)!/(l-2)!) Phi/(k S_K)^2/s2 (CLASS, dimensionful).
            sld   = jnp.sqrt(jnp.clip(q2 - K*lf**2, 1.e-30, None))
            dPhi  = sld*Phi_lm1 - (lf+1.)*cotK*Phi_l
            d2Phi = -2.*cotK*dPhi + (lf*(lf+1.)/sinK**2 - q2 + K)*Phi_l
            mask  = x_eff >= xmin_l

            r0 = jnp.where(mask, Phi_l, 0.)
            r1 = jnp.where(mask, dPhi, 0.)/k_axis
            r2 = jnp.where(mask, 3.*d2Phi/k_axis**2 + Phi_l, 0.)/(2.*s2)
            eps_factor = jnp.sqrt(3./8.*(lf+2.)*(lf+1.)*lf*(lf-1.))
            rE = eps_factor/s2 * jnp.where(mask, Phi_l/(k_axis*sinK)**2, 0.)

            transferT = jnp.sum(SW0*r0 + SW1*r1 + SW2*r2, axis=0)   # (Nk,)
            transferE = jnp.sum(SWE*rE, axis=0)

            clTT = jnp.sum(wk_prim*transferT**2)
            clTE = jnp.sum(wk_prim*transferT*transferE)
            clEE = jnp.sum(wk_prim*transferE**2)

            # Advance l -> l+1; clamp evanescent growth, terminate closed modes at l >= nu.
            slp_arg = q2 - K*(lf+1.)**2
            slpd = jnp.sqrt(jnp.clip(slp_arg, 1.e-30, None))
            Phi_next = jnp.where(slp_arg > term_tol,
                                 jnp.clip(((2.*lf+1.)*cotK*Phi_l - sld*Phi_lm1)/slpd, -1.e10, 1.e10),
                                 0.)
            return (Phi_l, Phi_next), jnp.stack((clTT, clTE, clEE))

        # Chunked scan over l; jax.checkpoint per chunk bounds reverse-AD residency.
        CHUNK = 64
        n = self.curv_ells.shape[0]
        npad = (-n) % CHUNK
        lf_all = jnp.concatenate((self.curv_ells.astype(jnp.float64),
                                  self.curv_ells[-1] + 1. + jnp.arange(npad, dtype=jnp.float64)))
        xmin_all = jnp.concatenate((self.curv_xmin, jnp.full((npad,), self.curv_xmin[-1])))
        xs = (lf_all.reshape(-1, CHUNK), xmin_all.reshape(-1, CHUNK))

        def chunk_body(carry, xs_chunk):
            return lax.scan(step, carry, xs_chunk)

        _, outs = lax.scan(jax.checkpoint(chunk_body), (Phi1, Phi2), xs)
        cls = outs.reshape(-1, 3)[:n]
        return cls[:, 0], cls[:, 1], cls[:, 2]