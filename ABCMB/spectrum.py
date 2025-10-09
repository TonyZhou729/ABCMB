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

bessel_l_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/l.txt", dtype="int"))
bessel_x_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/x.txt"))
bessel_stop_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/jl_stop.txt"))

# 2D arrays of tabulated spherical functions over l and x axes.
bessel_phi0_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/phi0.txt"))
bessel_phi1_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/phi1.txt"))
bessel_phi2_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/phi2.txt"))
bessel_epsilon_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/epsilon.txt"))

try:
    gpus = jax.devices('gpu')
    bessel_l_tab = jax.device_put(
        bessel_l_tab, device=gpus[0])
    bessel_x_tab = jax.device_put(
        bessel_x_tab, device=gpus[0])
    bessel_stop_tab = jax.device_put(
        bessel_stop_tab, device=gpus[0])
    bessel_phi0_tab = jax.device_put(
        bessel_phi0_tab, device=gpus[0])
    bessel_phi1_tab = jax.device_put(
        bessel_phi1_tab, device=gpus[0])
    bessel_phi2_tab = jax.device_put(
        bessel_phi2_tab, device=gpus[0])
    bessel_epsilon_tab = jax.device_put(
        bessel_epsilon_tab, device=gpus[0])
except: 
    pass


def phi0(i, x):
    """
    Compute spherical Bessel function φ₀.

    First integer argument i indicates we are computing this for l = bessel_l_tab[i]
    float argument x is the argument of the bessel function.

    Parameters:
    -----------
    i : int
        Index into bessel_l_tab for multipole ℓ
    x : float
        Argument of spherical Bessel function

    Returns:
    --------
    float
        φ₀(x) for multipole ℓ = bessel_l_tab[i]

    Notes:
    ------
    1312.2697 Eq. (3.19a)
    """
    # Annoyingly the following line is not jit safe...
    # For now I am passing in the idx corresponding to the theoretically desired l.
    #idx = jnp.where(bessel_l_tab == ell)[0][0].item()
    return tools.fast_interp(x, bessel_x_tab.min(), bessel_x_tab.max(), bessel_phi0_tab[:, i])

def phi1(i, x):
    """
    Compute spherical Bessel function φ₁.

    Parameters:
    -----------
    i : int
        Index into bessel_l_tab for multipole ℓ
    x : float
        Argument of spherical Bessel function

    Returns:
    --------
    float
        φ₁(x) for multipole ℓ = bessel_l_tab[i]

    Notes:
    ------
    1312.2697 Eq. (3.19a)
    """
    #idx = jnp.where(bessel_l_tab == ell)[0][0].item()
    return tools.fast_interp(x, bessel_x_tab.min(), bessel_x_tab.max(), bessel_phi1_tab[:, i])

def phi2(i, x):
    """
    Compute spherical Bessel function φ₂.

    Parameters:
    -----------
    i : int
        Index into bessel_l_tab for multipole ℓ
    x : float
        Argument of spherical Bessel function

    Returns:
    --------
    float
        φ₂(x) for multipole ℓ = bessel_l_tab[i]

    Notes:
    ------
    1312.2697 Eq. (3.19a)
    """
    #idx = jnp.where(bessel_l_tab == ell)[0][0].item()
    return tools.fast_interp(x, bessel_x_tab.min(), bessel_x_tab.max(), bessel_phi2_tab[:, i])

def epsilon(i, x):
    """
    Compute polarization coupling function ε.

    Parameters:
    -----------
    i : int
        Index into bessel_l_tab for multipole ℓ
    x : float
        Argument of coupling function

    Returns:
    --------
    float
        ε(x) for multipole ℓ = bessel_l_tab[i]

    Notes:
    ------
    1312.2697 Eq. (3.19b)
    """
    #idx = jnp.where(bessel_l_tab == ell)[0][0].item()
    return tools.fast_interp(x, bessel_x_tab.min(), bessel_x_tab.max(), bessel_epsilon_tab[:, i])

class SpectrumSolver(eqx.Module):
    """
    CMB angular power spectrum computation.

    Computes temperature and polarization angular power spectra by
    integrating transfer functions over wavenumber and time.

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
    lensing : bool

    k_pivot    : float = 0.05 # In 1/Mpc
    switch_sw  : float = 1.
    switch_isw : float = 1.
    switch_dop : float = 1.
    switch_pol : float = 1.

    def __init__(self,
                 ellmin=2,
                 ellmax=2500,
                 lensing=True,
                 k_pivot=0.05,
                 switch_sw=1,
                 switch_isw=1,
                 switch_dop=1,
                 switch_pol=1):

        self.ells = jnp.arange(ellmin, ellmax+1)
        ell_idx_min = jnp.where(bessel_l_tab<=ellmin)[0][-1]
        ell_idx_max = jnp.where(bessel_l_tab>=ellmax)[0][0]
        self.ells_indices = jnp.arange(ell_idx_min, ell_idx_max+1)
        
        self.lensing = lensing

        self.k_pivot    = k_pivot
        self.switch_sw  = switch_sw
        self.switch_isw = switch_isw
        self.switch_dop = switch_dop
        self.switch_pol = switch_pol

    def primordial_spectrum(self, k, BG):
        """
        Compute primordial curvature power spectrum.

        Parameters:
        -----------
        k : float or array
            Wavenumber (units: Mpc^{-1})
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float or array
            Primordial power spectrum P_R(k), units Mpc^3
        """
        return BG.params['A_s']*(k/self.k_pivot)**(BG.params['n_s']-1.) * (2*jnp.pi**2/k**3)

    def Pk_lin(self, k, z, PT, BG):
        """
        Compute linear matter power spectrum at wavenumbers k and redshift z.

        Parameters
        ----------
        k : float or array
            Wavenumber (Mpc^{-1})
        z : float
            Redshift to evaluate.
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : cosmology.Background
            Background cosmology module

        Returns
        -------
        float or array
            Linear matter power spectrum P(k, z), units Mpc^3
        """

        lna = -jnp.log(1.+z)
    
        # vmapped interpolation over Nk (columns of the 2D arrays)
        interp_over_lna = jax.vmap(
            lambda y: jnp.interp(lna, PT.lna, y),
            in_axes=1  # loop over columns
        )

        delta_cdm_lna = interp_over_lna(PT.delta_cdm)  # shape (Nk,)
        delta_b_lna   = interp_over_lna(PT.delta_b)    # shape (Nk,)

        # now interpolate over k
        delta_cdm = jnp.interp(k, PT.k, delta_cdm_lna)
        delta_b   = jnp.interp(k, PT.k, delta_b_lna)

        # total matter overdensity
        delta_m = (
            BG.params['omega_b']   * delta_b +
            BG.params['omega_cdm'] * delta_cdm
        ) / BG.params['omega_m']

        return delta_m**2 * self.primordial_spectrum(k, BG)

    def lensing_power_spectrum(self, k, lna, PT, BG):
        """
        Computes the lensing power spectrum at wavenumbers k and redshift z.
        Eq.(3.15) in astro-ph/0601594

        Parameters
        ----------
        k : float or array
            Wavenumber (Mpc^{-1})
        lna : float
            Scale factor
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : cosmology.Background
            Background cosmology module

        Returns
        -------
        float or array
            Lensing matter power spectrum P(k, z), dimensionless.
        """
        a = jnp.exp(lna)
        z = 1./a - 1.
        aH = BG.aH(lna)

        Omega_m = BG.params["omega_m"]/BG.params["h"]**2
        Omega_L = BG.params["omega_Lambda"]/BG.params["h"]**2

        # Matter fraction over time after equality. 1 at early times and becomes Om0 today. 
        Om = (Omega_m * (1.+z)**3)/ ((Omega_m * (1.+z)**3) + Omega_L)

        Pk = self.Pk_lin(k, z, PT, BG) # Mpc^3

        return 9./8./jnp.pi**2 * Om**2 * aH**4 * Pk / k

    def lensing_Cl(self, ells, PT, BG):
        """
        Angular lensing power spectrum at multipole ell.

        IMPORTANT: Assumes Limber approximation throughout, even at ell=2.

        Eq.(3.14) in astro-ph/0601594, except shifts ell -> ell+1/2 to match CLASS.

        Parameters
        ----------
        ell : float or array
            Multipole
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : cosmology.Background
            Background cosmology module

        Returns
        -------
        float or array
            Angular lensing matter power spectrum Cl^phiphi, dimensionless.
        """

        coeff = 8.*jnp.pi**2/(ells+0.5)**3
        chi = lambda lna : BG.tau0 - BG.tau(lna)

        def integrand_func(lna):
            k = (ells+0.5)/chi(lna)
            window = (chi(BG.lna_rec) - chi(lna))/chi(BG.lna_rec)/chi(lna)
            res = chi(lna)/BG.aH(lna) * window**2 * self.lensing_power_spectrum(k, lna, PT, BG)
            return res

        lna_axis = jnp.linspace(BG.lna_rec, 0., 4000)
        integrand = vmap(integrand_func)(lna_axis)
        return coeff*jnp.trapezoid(integrand, lna_axis, axis=0)

    def lensed_Cls(self, ells, ClTT_unlensed, ClTE_unlensed, ClEE_unlensed, PT, BG):
        #beta = jnp.linspace(0., jnp.pi/16., 5000)
        #mu = jnp.cos(beta)
        mu = jnp.linspace(jnp.cos(jnp.pi/16.), 1., 1000)

        # Compute lensing Cl
        Clpp = self.lensing_Cl(ells, PT, BG)

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
        #sigma2     = Cgl[0] - Cgl
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
                - d00
            ), 
            axis=1
        )

        ksip = 1./4./jnp.pi * jnp.sum(
            (2.*ells+1)*ClEE_unlensed * (
                X022**2 * d22 \
                + 2*Cgl2*X132*X121*d31 \
                + Cgl2**2 * (X022_prime**2*d22 + X242*X220*d40) \
                - d22
            ), 
            axis=1
        )

        ksim = 1./4./jnp.pi * jnp.sum(
            (2.*ells+1)*ClEE_unlensed * (
                X022**2 * d2m2 \
                + Cgl2*(X121**2*d1m1 + X132**2*d3m3) \
                + 1./2.*Cgl2**2 * (2*X022_prime**2*d2m2 + X220**2*d00 + X242**2*d4m4) \
                - d2m2
            ), 
            axis=1
        )

        ksix = 1./4./jnp.pi * jnp.sum(
            (2.*ells+1)*ClTE_unlensed * (
                X022*X000*d02 \
                + Cgl2 * 2*X000_prime/jnp.sqrt(llp1) * (X121*d11 + X132*d3m1) \
                + 1./2.*Cgl2**2 * ((2*X022_prime*X000_prime+X220**2)*d20+X220*X242*dm24) \
                - d02
            ), 
            axis=1
        )
        
        #ClTT = 2.*jnp.pi * jnp.trapezoid(corTT[:, None]*d00*jnp.sin(beta)[:, None], beta, axis=0) # Integrand becomes (Nmu, Nells), result is Nells
        ClTT = 2.*jnp.pi * jnp.trapezoid(ksi[:, None]*d00, mu, axis=0) + ClTT_unlensed
        ClTE = 2.*jnp.pi * jnp.trapezoid(ksix[:, None]*d20, mu, axis=0) + ClTE_unlensed
        ClEE = 1./2. * 2.*jnp.pi * jnp.trapezoid(ksip[:, None]*d22+ksim[:, None]*d2m2, mu, axis=0) + ClEE_unlensed

        return (ClTT, ClTE, ClEE)

    # def get_Cl(self, PT, BG):
    #     """
    #     Compute angular power spectra for multiple multipoles.

    #     Parameters:
    #     -----------
    #     idxs : array
    #         Indices into bessel_l_tab for desired multipoles
    #     PT : perturbations.PerturbationTable
    #         Perturbation evolution table
    #     BG : cosmology.Background
    #         Background cosmology module

    #     Returns:
    #     --------
    #     array
    #         Angular power spectra (C_ℓ^TT, C_ℓ^TE, C_ℓ^EE) for each ℓ
    #     """
    #     ells = bessel_l_tab[self.ells_indices]
    #     Cls_raw = vmap(lambda idx : self.Cl_one_ell(idx, PT, BG))(self.ells_indices)
    #     return ells, Cls_raw
    #     # Cubic interpolate onto 

    #     return 0.

    def get_Cl(self, PT, BG):
        """
        Compute angular power spectra for multiple multipoles using lax.scan.
        """

        def scan_fun(_, idx):
            cltt, clte, clee = self.Cl_ee_only(idx, PT, BG)
            return None, jnp.array([cltt, clte, clee])

        _, Cls_raw = lax.scan(scan_fun, None, self.ells_indices)

        # Cubic spline for smooth Cl over user requested ells

        tt_raw = Cls_raw[:, 0]
        te_raw = Cls_raw[:, 1]
        ee_raw = Cls_raw[:, 2]

        ells = bessel_l_tab[self.ells_indices]
        tt_unlensed = CubicSpline(ells, tt_raw, check=False)(self.ells)
        te_unlensed = CubicSpline(ells, te_raw, check=False)(self.ells)
        ee_unlensed = CubicSpline(ells, ee_raw, check=False)(self.ells)

        def get_lensed_Cls():
            return self.lensed_Cls(self.ells, tt_unlensed, te_unlensed, ee_unlensed, PT, BG)

        def get_unlensed_Cls():
            return (tt_unlensed, te_unlensed, ee_unlensed)

        #return (tt_unlensed, te_unlensed, ee_unlensed)
        return lax.cond(
            self.lensing,
            get_lensed_Cls,
            get_unlensed_Cls
        )
        #return get_lensed_Cls()

    def get_Cl_vmap(self, PT, BG):
        """
        Compute angular power spectra for multiple multipoles using lax.scan.
        """

        tt_raw, te_raw, ee_raw = vmap(self.Cl_ee_only, in_axes=(0, None, None))(self.ells_indices, PT, BG)

        ells = bessel_l_tab[self.ells_indices]
        tt_unlensed = CubicSpline(ells, tt_raw, check=False)(self.ells)
        te_unlensed = CubicSpline(ells, te_raw, check=False)(self.ells)
        ee_unlensed = CubicSpline(ells, ee_raw, check=False)(self.ells)

        def get_lensed_Cls():
            return self.lensed_Cls(self.ells, tt_unlensed, te_unlensed, ee_unlensed, PT, BG)

        def get_unlensed_Cls():
            return (tt_unlensed, te_unlensed, ee_unlensed)

        #return (tt_unlensed, te_unlensed, ee_unlensed)
        return lax.cond(
            self.lensing,
            get_lensed_Cls,
            get_unlensed_Cls
        )
        #return get_lensed_Cls()

    def Cl_one_ell(self, idx, PT, BG):
        """
        Computes angular power spectrum for single multipole.

        Integrates transfer functions over wavenumber.

        Parameters:
        -----------
        idx : int
            Index into bessel_l_tab for multipole ℓ
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        tuple
            (C_ℓ^TT, C_ℓ^TE, C_ℓ^EE) angular power spectra
        """
        # Beyond this k point the bessel function vanishes exponentially.
        k_cut_small = 0.95*bessel_l_tab[idx]/BG.rA_rec

        # The upperbound of the integral is given by the multipole cut approximation in arxiv:1312.2697
        # For now, the integration axis is chosen to be a logspaced grid, from kmin to kmin+kcut.
        # This is because for k>kmin, the integrand ~jl^2 which experiences asymptotic damping for larger k's.
        # The peak values of the envelope drop by a few orders of magnitude within 3-4 peaks or so, so its
        # only really important to have high resolution near kmin. 
        k_T0_axis = jnp.geomspace(k_cut_small, k_cut_small+0.1, 400) 
        lna_axis = PT.lna

        ### TRANSFER FUNCTION ###
        # Background quantities, all Nlna 1D vectors
        tau0 = BG.tau0
        tau = BG.tau(lna_axis)
        g   = BG.visibility(lna_axis)
        g_prime = vmap(grad(BG.visibility))(lna_axis) # Derivative of g w.r.t. lna
        aH  = BG.aH(lna_axis)
        expmkappa = BG.expmkappa(lna_axis)
        aH_dot = BG.aH_prime(lna_axis) * aH # Derivative of aH w.r.t. conformal time tau.

        g        = g[:, None]
        g_prime  = g_prime[:, None]
        aH       = aH[:, None]
        expmkappa    = expmkappa[:, None]
        aH_dot   = aH_dot[:, None]

        # Perturbations, all (Nk, Nlna) 2D vectors
        interp_column = lambda col : jnp.interp(jnp.log10(k_T0_axis), jnp.log10(PT.k), col)

        # Found that this is much much faster than RegularGridInterpolator
        delta_g       = vmap(interp_column, in_axes=0, out_axes=0)(PT.delta_g)
        theta_b       = vmap(interp_column, in_axes=0, out_axes=0)(PT.theta_b)
        theta_b_prime = vmap(interp_column, in_axes=0, out_axes=0)(PT.theta_b_prime)
        sigma_g       = vmap(interp_column, in_axes=0, out_axes=0)(PT.sigma_g)
        Gg0           = vmap(interp_column, in_axes=0, out_axes=0)(PT.Gg0)
        Gg2           = vmap(interp_column, in_axes=0, out_axes=0)(PT.Gg2)
        eta           = vmap(interp_column, in_axes=0, out_axes=0)(PT.metric_eta)
        eta_prime     = vmap(interp_column, in_axes=0, out_axes=0)(PT.metric_eta_prime)
        alpha         = vmap(interp_column, in_axes=0, out_axes=0)(PT.metric_alpha)
        alpha_prime   = vmap(interp_column, in_axes=0, out_axes=0)(PT.metric_alpha_prime)
        
        #sourceT0 = self.switch_sw * g * (delta_g/4. + aH*alpha_prime) 
        #sourceT0 = 1.

        # Source terms
        # TODO: fix ISW term
        sourceT0 = self.switch_sw * g * (delta_g/4. + aH*alpha_prime) \
                + self.switch_isw * (
                    g * (eta - aH*alpha_prime - 2.*aH*alpha) \
                    + 2.*expmkappa * (aH*eta_prime - aH_dot*alpha - aH**2*alpha_prime)
                ) \
                + self.switch_dop * (
                    aH * (g*((theta_b_prime / k_T0_axis**2) + alpha_prime) \
                    + g_prime*((theta_b / k_T0_axis**2) + alpha))
                )
        #sourceT0 = 0.

        sourceT1 = self.switch_isw * expmkappa * \
                ((aH*alpha_prime + 2.*aH*alpha - eta) * k_T0_axis)
        #sourceT1 = 0.

        sourceT2 = self.switch_pol * g * (2*sigma_g + Gg0 + Gg2) / 8.

        sourceE  = jnp.sqrt(6) * sourceT2

        # Bessel functions
        chiT0 = jnp.outer(tau0-tau, k_T0_axis)

        transferT0 = jnp.trapezoid(
            sourceT0 / aH * phi0(idx, chiT0),
            lna_axis, axis=0
        )

        transferT1 = jnp.trapezoid(
            sourceT1 / aH * phi1(idx, chiT0),
            lna_axis, axis=0
        )

        transferT2 = jnp.trapezoid(
            sourceT2 / aH * phi2(idx, chiT0),
            lna_axis, axis=0
        )

        transferE = jnp.trapezoid(
            sourceE / aH * epsilon(idx, chiT0),
            lna_axis, axis=0
        )

        del chiT0

        transferT = transferT0 + transferT1 + transferT2
        ### END OF TRANSFER FUNCTION ###

        ### LINE OF SIGHT INTEGRAL ###
        integrandTT = 4.*jnp.pi * BG.params['A_s'] * (k_T0_axis/self.k_pivot)**(BG.params['n_s']-1.) * transferT**2 / k_T0_axis
        integrandTE = 4.*jnp.pi * BG.params['A_s'] * (k_T0_axis/self.k_pivot)**(BG.params['n_s']-1.) * transferT*transferE / k_T0_axis
        integrandEE = 4.*jnp.pi * BG.params['A_s'] * (k_T0_axis/self.k_pivot)**(BG.params['n_s']-1.) * transferE**2 / k_T0_axis
        #return k_T0_axis, (integrandTT, integrandTE, integrandEE)
        return (jnp.trapezoid(integrandTT, k_T0_axis), jnp.trapezoid(integrandTE, k_T0_axis), jnp.trapezoid(integrandEE, k_T0_axis))

    def Cl_ee_only(self, idx, PT, BG):
        # Beyond this k point the bessel function vanishes exponentially.
        k_cut_small = 0.9*bessel_l_tab[idx]/BG.rA_rec

        k_E_axis = jnp.geomspace(k_cut_small, k_cut_small+0.11, 5000) 
        lna_axis = PT.lna

        tau0 = BG.tau0
        tau = BG.tau(lna_axis)
        aH = BG.aH(lna_axis)
        g = BG.visibility(lna_axis)

        tau = tau[:, None]
        aH = aH[:, None]
        g = g[:, None]

        interp_column = lambda col : jnp.interp(jnp.log10(k_E_axis), jnp.log10(PT.k), col)

        sigma_g       = vmap(interp_column, in_axes=0, out_axes=0)(PT.sigma_g)
        Gg0           = vmap(interp_column, in_axes=0, out_axes=0)(PT.Gg0)
        Gg2           = vmap(interp_column, in_axes=0, out_axes=0)(PT.Gg2)

        sourceT2 = self.switch_pol * g * (2*sigma_g + Gg0 + Gg2) / 8.

        sourceE  = jnp.sqrt(6) * sourceT2

        chi = jnp.outer(tau0-tau, k_E_axis)

        transferE = jnp.trapezoid(
            sourceE / aH * epsilon(idx, chi),
            lna_axis, axis=0
        )

        del chi

        ### END OF TRANSFER FUNCTION ###

        ### LINE OF SIGHT INTEGRAL ###
        integrandEE = 4.*jnp.pi * BG.params['A_s'] * (k_E_axis/self.k_pivot)**(BG.params['n_s']-1.) * transferE**2 / k_E_axis

        return ((0, 0, jnp.trapezoid(integrandEE, k_E_axis)))

    ### OLD CODE ###

    #@jit
    def Cl_one_ell_with_loops(self, ell_idx, PT, BG):
        k_cut_small = bessel_l_tab[ell_idx]/BG.rA_rec
        k_T0_axis = jnp.linspace(k_cut_small, k_cut_small+0.075, 500)
        k_T1_axis = jnp.linspace(k_cut_small, k_cut_small+0.02, 130)
        k_T2_axis = jnp.linspace(k_cut_small, k_cut_small+0.075, 500)
        k_E_axis  = jnp.linspace(k_cut_small, k_cut_small+0.055, 360)

        #lna_max = BG.lna_visibility_stop # It'd be ideal to implement this time cut.
        #lna_max = PT.lna[-1]

        #This is slow
        # lna_max = jnp.where(
        #     bessel_l_tab[ell_idx] <= 400,
        #     BG.lna_visibility_stop,
        #     PT.lna[-1]
        # )

        # N_steps = jnp.where(
        #     bessel_l_tab[ell_idx] <= 400,
        #     43,
        #     PT.lna.size
        # )


        ###################
        ### TRANSFER T0 ###
        ################### 
        
        # def scan_T0(carry, k):
        #     val = self.integrate_trapezoid_while_loop(ell_idx, k, self.integrand_T0, PT, BG)
        #     return carry, val
        
        # _, transferT0 = lax.scan(scan_T0, None, k_T0_axis)

        # Wow vmap works here and is twice as fast as scan!!
        f = lambda k : self.integrate_trapezoid_while_loop(ell_idx, k, self.integrand_T0, PT, BG)
        #f = lambda k : self.integrate_trapezoid_trapz(ell_idx, k, self.integrand_T0, N_steps, PT, BG)
        #f = lambda k : self.integrate_trapezoid_scan(ell_idx, k, self.integrand_T0, PT, BG)
        #f = lambda k : self.integrate_trapezoid_vmap_scan(ell_idx, k, self.integrand_T0, PT, BG)
        transferT0 = vmap(f)(k_T0_axis)

        # ###################
        # ### TRANSFER T1 ###
        # ################### 
        
        # # def scan_T1(carry, k):
        # #     val = self.integrate_trapezoid_while_loop(ell_idx, k, self.integrand_T1, PT, BG)
        # #     return carry, val
        
        # # _, transferT1 = lax.scan(scan_T1, None, k_T0_axis)
        # f = lambda k : self.integrate_trapezoid_while_loop(ell_idx, k, self.integrand_T1, lna_max, PT, BG)
        # transferT1 = vmap(f)(k_T0_axis)

        # ###################
        # ### TRANSFER T2 ###
        # ################### 
        
        # # def scan_T2(carry, k):
        # #     val = self.integrate_trapezoid_while_loop(ell_idx, k, self.integrand_T2, PT, BG)
        # #     return carry, val
        
        # # _, transferT2 = lax.scan(scan_T2, None, k_T0_axis)
        # f = lambda k : self.integrate_trapezoid_while_loop(ell_idx, k, self.integrand_T2, lna_max, PT, BG)
        # transferT2 = vmap(f)(k_T0_axis)

        # ###################
        # ### TRANSFER E ###
        # ################### 
        
        # # def scan_E(carry, k):
        # #     val = self.integrate_trapezoid_while_loop(ell_idx, k, self.integrand_E, PT, BG)
        # #     return carry, val
        
        # # _, transferE = lax.scan(scan_E, None, k_T0_axis)
        # f = lambda k : self.integrate_trapezoid_while_loop(ell_idx, k, self.integrand_E, lna_max, PT, BG)
        # transferE = vmap(f)(k_T0_axis)

        transferT = transferT0 #+ transferT1 + transferT2
        #transferT = transferT0 + tools.fast_interp(k_T0_axis, k_T1_axis[0], k_T1_axis[-1], transferT1) + tools.fast_interp(k_T0_axis, k_T2_axis[0], k_T2_axis[-1], transferT2)
        #transferE = tools.fast_interp(k_T0_axis, k_E_axis[0], k_E_axis[-1], transferE)

        integrandTT = 4.*jnp.pi * BG.params['A_s'] * (k_T0_axis/self.k_pivot)**(BG.params['n_s']-1.) * transferT**2 / k_T0_axis
        #integrandTE = 4.*jnp.pi * BG.params['A_s'] * (k_T0_axis/self.k_pivot)**(BG.params['n_s']-1.) * transferT*transferE / k_T0_axis
        #integrandEE = 4.*jnp.pi * BG.params['A_s'] * (k_T0_axis/self.k_pivot)**(BG.params['n_s']-1.) * transferE**2 / k_T0_axis
        #return (jnp.trapezoid(integrandTT, k_T0_axis), jnp.trapezoid(integrandTE, k_T0_axis), jnp.trapezoid(integrandEE, k_T0_axis))
        return (jnp.trapezoid(integrandTT, k_T0_axis), 0., 0.)

    def integrate_trapezoid_scan(self, ell_idx, k, integrand, PT, BG):
        """
        Trapezoid rule using lax.scan, computing integrand at each step.
        No precomputation with vmap. Integrates over N_steps of PT.lna.
        """

        dlna = PT.lna[1] - PT.lna[0]  # Assumes uniform spacing

        def scan_body(state, i):
            acc, f_prev = state
            f_next = integrand(ell_idx, k, i + 1, PT, BG)
            area = 0.5 * dlna * (f_prev + f_next)
            new_acc = acc + area
            return (new_acc, f_next), None

        # Initial state
        f0 = integrand(ell_idx, k, 0, PT, BG)
        init_state = (0.0, f0)

        (final_acc, _), _ = lax.scan(scan_body, init_state, jnp.arange(PT.lna.size))
        return final_acc

    def integrate_trapezoid_vmap_scan(self, ell_idx, k, integrand, PT, BG):
        """
        Trapezoidal integrator using lax.scan over a fixed number of steps N_steps.
        integrand(ell_idx, k, lna_idx, PT, BG) returns the value at each step.
        PT.lna is assumed to be uniformly spaced.
        """
        dlna = PT.lna[1] - PT.lna[0]  # assumes uniform spacing

        # Precompute all needed function values
        indices = jnp.arange(PT.lna.size)
        f_vals = vmap(lambda i: integrand(ell_idx, k, i, PT, BG))(indices)

        def scan_body(acc, i):
            area = 0.5 * dlna * (f_vals[i] + f_vals[i + 1])
            return acc + area, None

        integral, _ = lax.scan(scan_body, 0.0, jnp.arange(PT.lna.size))
        return integral

    def integrate_trapezoid_trapz(self, ell_idx, k, integrand, N_steps, PT, BG):
        """
        Integrate using jnp.trapz over N_steps of PT.lna and integrand values.
        """
        indices = jnp.arange(N_steps + 1)
        lna_slice = PT.lna[indices]
        f_vals = vmap(lambda i: integrand(ell_idx, k, i, PT, BG))(indices)
        
        return jnp.trapezoid(f_vals, lna_slice)

    def integrate_trapezoid_while_loop(self, ell_idx, k, integrand, PT, BG):
        """
        A while loop trapezoid rule integrator.
        Specifically used for integrating across lna for transfer function, given k.

        integrand should take (ell_idx, k, lna_idx) and return the integrand.

        Returns the transfer function for one ell and k value.
        """
        """
        lna_vals = jnp.linspace(BG.lna_transfer_start, 0.0, 3000)
        lna_max = lna_vals[jnp.argmin((BG.tau(lna_vals)-BG.tau0+bessel_stop_tab[ell_idx]/k)**2)]

        def stop_condition(lna_idx):
            #Stop integrating when argument is past the ell of the bessel function.
            # Found that run time is heavily dependent on this stop cond.
            #return k*(BG.tau0-BG.tau(PT.lna[lna_idx])) > 0.9*bessel_l_tab[ell_idx]
            #return k*(BG.tau0-BG.tau(PT.lna[lna_idx])) > bessel_stop_tab[ell_idx]
            #return PT.lna[lna_idx] < -6.6
            return True
            #return PT.lna[lna_idx] < lna_max
        """
        def cond_fun(state):
            lna_idx, _, _ = state
            return jnp.logical_and(lna_idx+1 < PT.lna.size, k*(BG.tau0-BG.tau(PT.lna[lna_idx])) > bessel_stop_tab[ell_idx])
            #return lna_idx+1 < PT.lna.size
            #return PT.lna[lna_idx] < -6.6
            #return k*(BG.tau0-BG.tau(PT.lna[lna_idx])) > bessel_stop_tab[ell_idx]

        def body_fun(state):
            lna_idx, acc, f_prev = state
            f_next = integrand(ell_idx, k, lna_idx+1, PT, BG)
            area = 0.5 * (PT.lna[1]-PT.lna[0]) * (f_prev + f_next)
            return (lna_idx+1, acc + area, f_next)

        # Precompute initial f0
        f0 = integrand(ell_idx, k, 0, PT, BG)
        init_state = (0, 0.0, f0)

        _, integral, _ = lax.while_loop(cond_fun, body_fun, init_state)
        return integral


    def integrand_T0(self, ell_idx, k, lna_idx, PT, BG):
        """
        Compute temperature source function integrand.

        Calculates the integrand for SW, ISW, and Doppler contributions.

        Parameters:
        -----------
        ell_idx : int
            Index into bessel_l_tab for multipole ℓ
        k : float
            Wavenumber (units: Mpc^{-1})
        lna_idx : int
            Index into time grid
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Temperature source integrand
        """
        # Background
        lna = PT.lna[lna_idx] # Current scale factor
        chi = k*(BG.tau0-BG.tau(lna)) # Argument of phi0
        aH = BG.aH(lna)
        aH_dot = BG.aH_prime(lna) * aH
        g = BG.visibility(lna)
        g_prime = grad(BG.visibility)(lna)
        kappa = BG.kappa(lna)

        # Interpolate perturbation solutions
        # delta_g = jnp.interp(k, PT.k, PT.delta_g[:, lna_idx])
        # theta_b = jnp.interp(k, PT.k, PT.theta_b[:, lna_idx])
        # theta_b_prime = jnp.interp(k, PT.k, PT.theta_b_prime[:, lna_idx])
        # alpha = jnp.interp(k, PT.k, PT.metric_alpha[:, lna_idx])
        # alpha_prime = jnp.interp(k, PT.k, PT.metric_alpha_prime[:, lna_idx])
        # eta = jnp.interp(k, PT.k, PT.metric_eta[:, lna_idx])
        # eta_prime = jnp.interp(k, PT.k, PT.metric_eta_prime[:, lna_idx])

        # Faster with fast_interp
        logk = jnp.log10(k)
        logk_axis = jnp.log10(PT.k)
        delta_g = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.delta_g[:, lna_idx])
        theta_b = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.theta_b[:, lna_idx])
        theta_b_prime = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.theta_b_prime[:, lna_idx])
        alpha = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.metric_alpha[:, lna_idx])
        alpha_prime = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.metric_alpha_prime[:, lna_idx])
        eta = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.metric_eta[:, lna_idx])
        eta_prime = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.metric_eta_prime[:, lna_idx])

        ### Source Function ###
        sourceT0 = self.switch_sw * g * (delta_g/4. + aH*alpha_prime) \
                    + self.switch_isw * (
                    g * ( eta - aH*alpha_prime - 2.*aH*alpha ) \
                    + 2.*jnp.exp(-kappa) * ( aH*eta_prime - aH_dot*alpha - aH**2*alpha_prime )
                    ) \
                    + self.switch_dop * (
                    aH * (g*(theta_b_prime/k**2 + alpha_prime) + g_prime*(theta_b/k**2+alpha))
                    )
        #sourceT0 = self.switch_sw * g * (delta_g/4. + aH*alpha_prime)

        #return phi0(ell_idx, chi) / aH
        return sourceT0 * phi0(ell_idx, chi) / aH

    def integrand_T1(self, ell_idx, k, lna_idx, PT, BG):
        """
        Compute ISW temperature source function integrand.

        Parameters:
        -----------
        ell_idx : int
            Index into bessel_l_tab for multipole ℓ
        k : float
            Wavenumber (units: Mpc^{-1})
        lna_idx : int
            Index into time grid
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            ISW temperature source integrand
        """
        # Background
        lna = PT.lna[lna_idx] # Current scale factor
        chi = k*(BG.tau0-BG.tau(lna)) # Argument of phi0
        aH = BG.aH(lna)
        kappa = BG.kappa(lna)

        # Interpolate perturbation solutions
        logk = jnp.log10(k)
        logk_axis = jnp.log10(PT.k)
        alpha = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.metric_alpha[:, lna_idx])
        alpha_prime = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.metric_alpha_prime[:, lna_idx])
        eta = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.metric_eta[:, lna_idx])

        ### Source Function ###
        sourceT1 = self.switch_isw * jnp.exp(-kappa) * (aH*alpha_prime + 2.*aH*alpha - eta) * k

        return sourceT1 * phi1(ell_idx, chi) / aH
    
    def integrand_T2(self, ell_idx, k, lna_idx, PT, BG):
        """
        Compute polarization temperature source function integrand.

        Parameters:
        -----------
        ell_idx : int
            Index into bessel_l_tab for multipole ℓ
        k : float
            Wavenumber (units: Mpc^{-1})
        lna_idx : int
            Index into time grid
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Polarization temperature source integrand
        """
        # Background
        lna = PT.lna[lna_idx] # Current scale factor
        chi = k*(BG.tau0-BG.tau(lna)) # Argument of phi0
        aH = BG.aH(lna)
        g = BG.visibility(lna)

        # Interpolate perturbation solutions
        logk = jnp.log10(k)
        logk_axis = jnp.log10(PT.k)
        Fg2 = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.Fg2[:, lna_idx])
        Gg0 = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.Gg0[:, lna_idx])
        Gg2 = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.Gg2[:, lna_idx])

        ### Source Function ###
        sourceT2 = self.switch_pol * g * (Fg2+Gg0+Gg2) / 8.

        return sourceT2 * phi2(ell_idx, chi) / aH
    
    def integrand_E(self, ell_idx, k, lna_idx, PT, BG):
        """
        Compute E-mode polarization source function integrand.

        Parameters:
        -----------
        ell_idx : int
            Index into bessel_l_tab for multipole ℓ
        k : float
            Wavenumber (units: Mpc^{-1})
        lna_idx : int
            Index into time grid
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            E-mode polarization source integrand
        """
        # Background
        lna = PT.lna[lna_idx] # Current scale factor
        chi = k*(BG.tau0-BG.tau(lna)) # Argument of phi0
        aH = BG.aH(lna)
        g = BG.visibility(lna)

        # Interpolate perturbation solutions
        logk = jnp.log10(k)
        logk_axis = jnp.log10(PT.k)
        Fg2 = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.Fg2[:, lna_idx])
        Gg0 = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.Gg0[:, lna_idx])
        Gg2 = tools.fast_interp(logk, logk_axis[0], logk_axis[-1], PT.Gg2[:, lna_idx])

        ### Source Function ###
        sourceE = jnp.sqrt(6.) * self.switch_pol * g * (Fg2+Gg0+Gg2) / 8.

        return sourceE * epsilon(ell_idx, chi) / aH

    def stop_T0(self, ell_idx, k, lna_idx, PT, BG):
        return True

    def stop_T1T2E(self, ell_idx, k, lna_idx, PT, BG):
        # return jnp.where(
        #     bessel_l_tab[ell_idx] <= 400,
        #     PT.lna[lna_idx] < BG.lna_visibility_stop,
        #     True
        # )
        return PT.lna[lna_idx] < BG.lna_visibility_stop

    @jit
    def ClTT_loop(self, ell, PT, BG):
        ### OUTDATED FUNCTION ###
        
        k_cut_small = ell/BG.rA_rec
        k_cut_large = ell/BG.rA_rec + 0.15
        #deltak = 2.*jnp.pi/BG.tau0/10.
        #k_integral_axis = jnp.arange(k_cut_small, k_cut_large+deltak, deltak)
        k_integral_axis = jnp.linspace(k_cut_small, k_cut_large, 3500)
        vec_transfer_T0 = eqx.filter_vmap(
                partial(
                    self.scalar_T0_transfer,
                    ell=ell,
                    PT=PT,
                    BG=BG
                ),
                in_axes=0  # k will be vectorized over its first axis
            )
        vec_transfer_T1 = eqx.filter_vmap(
                partial(
                    self.scalar_T1_transfer,
                    ell=ell,
                    PT=PT,
                    BG=BG
                ),
                in_axes=0  # k will be vectorized over its first axis
            )
        transfer = vec_transfer_T0(k_integral_axis) #+ vec_transfer_T1(k_integral_axis)
        integrand = 4.*jnp.pi * BG.params['A_s'] * (k_integral_axis/self.k_pivot)**(BG.params['n_s']-1.) * transfer**2 / k_integral_axis
        return jnp.trapezoid(integrand, k_integral_axis)

    def scalar_T0_transfer(self, k, ell, PT, BG):
        ### OUTDATED FUNCTION ###
        """
        Computes the scalar transfer function T0 for the temperature anisotropy. 
        See Eqs.(3.1) & (3.5) of https://iopscience.iop.org/article/10.1088/1475-7516/2014/09/032/pdf

        Params:
            k : float
            ell : float
            PT : perturbations.PerturbationTable
            BG  : cosmology.Background
            firstx : float
        """
        tau0 = BG.tau0
        
        @jax.named_scope("integrand func")
        def integrand_func(lna, y, args):
            """
            The integrand over lna for the T0 CMB transfer function.
            We use Eq.(3.9) of arXiv:1312.2697, but in the SYNCHRONOUS GAUGE.
            This was read off in the source code for CLASS, in perturbations_sources()
            """

            # Background quantities
            tau = BG.tau(lna)
            g   = BG.visibility(lna)
            g_prime = grad(BG.visibility)(lna) # Derivative of g w.r.t. lna
            aH  = BG.aH(lna)
            kappa = BG.kappa(lna)
            chi = k*(tau0-tau)
            aH_dot = BG.aH_prime(lna) * aH # Derivative of aH w.r.t. conformal time tau.

            # Pertubations
            point = jnp.array([k, lna])
            delta_g = PT.delta_g(point)
            theta_b = PT.theta_b(point)
            theta_b_prime = PT.theta_b_prime(point)
            alpha = PT.metric_alpha(point)
            alpha_prime = PT.metric_alpha_prime(point)
            h_prime = PT.metric_h_prime(point)
            eta = PT.metric_eta(point)
            eta_prime = PT.metric_eta_prime(point)

            sourceT0 = self.switch_sw * g * (delta_g/4. + aH*alpha_prime) \
                     + self.switch_isw * (
                        g * ( eta - aH*alpha_prime - 2.*aH*alpha ) \
                        + 2.*jnp.exp(-kappa) * ( aH*eta_prime - aH_dot*alpha - aH**2*alpha_prime )
                     ) \
                     + self.switch_dop * (
                        aH/k**2 * (g*(theta_b_prime + alpha_prime*k**2) + g_prime*(theta_b+alpha*k**2))
                     )
            integrand  =  sourceT0 / aH * self.jl(ell, chi)
            return integrand[0]

        term = ODETerm(integrand_func)
        controller = PIDController(rtol=1.e-3, atol=1.e-6) # TODO: Try replacing with quadax, probably faster!

        # Translate to upper bound on lna, given x = k(tau_0 - tau)
        lna_stop_vals = jnp.linspace(BG.lna_transfer_start, 0.0, 1000)
        chi_stop_vals = k*(tau0-BG.tau(lna_stop_vals))
        lna_stop = lna_stop_vals[jnp.argmin((chi_stop_vals-self.jl_stop[ell])**2)]

        sol = diffeqsolve(
            term,
            solver=Kvaerno5(),
            t0=BG.lna_transfer_start,
            t1=lna_stop,
            dt0=1.e-2,     
            y0=0.,
            stepsize_controller=controller,
            max_steps = 10000
        )
        return sol.ys[0]

    def scalar_T1_transfer(self, k, ell, PT, BG):
        ### OUTDATED FUNCTION ###
        """
        Computes the scalar transfer function T1 for the temperature anisotropy. 
        See Eqs.(3.1) & (3.5) of https://iopscience.iop.org/article/10.1088/1475-7516/2014/09/032/pdf

        Params:
            k : float
            ell : float
            PT : perturbations.PerturbationTable
            BG  : cosmology.Background
        """
        tau0 = BG.tau0
        
        def integrand_func(lna, y, args):
            """
            The integrand over lna for the T1 CMB transfer function.
            We use Eq.(3.9) of arXiv:1312.2697, but in the SYNCHRONOUS GAUGE.
            This was read off in the source code for CLASS, in perturbations_sources()
            """

            # Background quantities
            tau = BG.tau(lna)
            aH  = BG.aH(lna)
            kappa = BG.kappa(lna)
            chi = k*(tau0-tau)

            # Pertubations
            point = jnp.array([k, lna])
            alpha = PT.metric_alpha(point)
            alpha_prime = PT.metric_alpha_prime(point)
            eta = PT.metric_eta(point)

            sourceT1 = self.switch_isw * jnp.exp(-kappa) * k * (
                aH*alpha_prime + 2.*aH*alpha - eta
            )

            integrand  =  sourceT1 / aH * self.jl_prime(ell, chi)
            return integrand[0]

        term = ODETerm(integrand_func)
        controller = PIDController(rtol=1.e-3, atol=1.e-6)

        # Translate to upper bound on lna, given x = k(tau_0 - tau)
        lna_stop_vals = jnp.linspace(BG.lna_transfer_start, 0.0, 1000)
        chi_stop_vals = k*(tau0-BG.tau(lna_stop_vals))
        lna_stop = lna_stop_vals[jnp.argmin((chi_stop_vals-self.jl_stop[ell])**2)]

        sol = diffeqsolve(
            term,
            solver=Kvaerno5(),
            t0=BG.lna_transfer_start,
            t1=lna_stop,
            dt0=1.e-2,     
            y0=0.,
            stepsize_controller=controller,
            max_steps = 10000
        )
        return sol.ys[0]

    def scalar_T2_transfer(self, k, ell, PT, BG):
        ### OUTDATED FUNCTION ###
        """
        Computes the scalar transfer function T2 for the temperature anisotropy. 
        See Eqs.(3.1) & (3.5) of https://iopscience.iop.org/article/10.1088/1475-7516/2014/09/032/pdf

        Params:
            k : float
            ell : float
            PT : perturbations.PerturbationTable
            BG  : cosmology.Background
        """
        tau0 = BG.tau0
        
        def integrand_func(lna, y, args):
            """
            The integrand over lna for the T1 CMB transfer function.
            We use Eq.(3.9) of arXiv:1312.2697, but in the SYNCHRONOUS GAUGE.
            This was read off in the source code for CLASS, in perturbations_sources()
            """

            # Background quantities
            tau = BG.tau(lna)
            aH  = BG.aH(lna)
            kappa = BG.kappa(lna)
            chi = k*(tau0-tau)

            # Pertubations
            point = jnp.array([k, lna])
            alpha = PT.metric_alpha(point)
            alpha_prime = PT.metric_alpha_prime(point)
            eta = PT.metric_eta(point)

            sourceT1 = self.switch_isw * jnp.exp(-kappa) * k * (
                aH*alpha_prime + 2.*aH*alpha - eta
            )

            integrand  =  sourceT1 / aH * self.jl_prime(ell, chi)
            return integrand[0]

        term = ODETerm(integrand_func)
        controller = PIDController(rtol=1.e-3, atol=1.e-6)

        # Translate to upper bound on lna, given x = k(tau_0 - tau)
        lna_stop_vals = jnp.linspace(BG.lna_transfer_start, 0.0, 1000)
        chi_stop_vals = k*(tau0-BG.tau(lna_stop_vals))
        lna_stop = lna_stop_vals[jnp.argmin((chi_stop_vals-self.jl_stop[ell])**2)]

        sol = diffeqsolve(
            term,
            solver=Kvaerno5(),
            t0=BG.lna_transfer_start,
            t1=lna_stop,
            dt0=1.e-2,     
            y0=0.,
            stepsize_controller=controller,
            max_steps = 10000
        )
        return sol.ys[0]

    @jit
    @jax.named_scope("ClTT diffrax")
    def ClTT_diffrax(self, ell, PT, BG):
        def integrand_func(k, y, args):
            transfer = self.scalar_T0_transfer(k, ell, PT, BG) + self.scalar_T1_transfer(k, ell, PT, BG)
            integrand = 4.*jnp.pi * BG.params['A_s'] * (k/self.k_pivot)**(BG.params['n_s']-1.) * transfer**2 / k
            return integrand
        
        term = ODETerm(integrand_func)
        controller = PIDController(rtol=1.e-2, atol=1.e-2)

        sol = diffeqsolve(
            term,
            solver=Kvaerno5(),
            t0=5.e-3,
            t1=0.5,
            dt0=1.e-3,     
            y0=0.,
            stepsize_controller=controller,
            max_steps = 30000
        )
        return sol.ys[0]

    #@jit
    def ClEE(self, ell, k_axis, lna_axis, Fg2, Gg0, Gg2, BG, A_s, n_s):
        
        Fg2_interp_func = RegularGridInterpolator((k_axis, lna_axis), Fg2)
        Gg0_interp_func = RegularGridInterpolator((k_axis, lna_axis), Gg0)
        Gg2_interp_func = RegularGridInterpolator((k_axis, lna_axis), Gg2)

        #k_integral_axis = jnp.linspace(k_axis[0], k_axis[-1], 5000)
        k_integral_axis = jnp.linspace(0.9, 2.0, 1000)*ell/BG.tau0 # Callin 2006 convention
        vec_source = eqx.filter_vmap(
                partial(
                    self.scalar_E_source,
                    ell=ell,
                    lna_axis=lna_axis,
                    Fg2_func=Fg2_interp_func,
                    Gg0_func=Gg0_interp_func,
                    Gg2_func=Gg2_interp_func,
                    BG=BG
                ),
                in_axes=0  # k will be vectorized over its first axis
            )
        sources = vec_source(k_integral_axis)
        integrand = 4.*jnp.pi * A_s * (k_integral_axis/self.k_pivot)**(n_s-1.) * sources**2 / k_integral_axis
        return jnp.trapezoid(integrand, k_integral_axis)

        # def integrand_func(k, y, args):
        #     E_source = self.scalar_E_source(ell, k, lna_axis, Fg2_interp_func, Gg0_interp_func, Gg2_interp_func, BG)
        #     PR = A_s #* (k/self.k_pivot)**(n_s-1.)
        #     return E_source**2 * PR / k

        # term = ODETerm(integrand_func)
        # controller = PIDController(rtol=1.e-3, atol=1.e-6)
        # sol = diffeqsolve(
        #     term,
        #     solver=Kvaerno5(),
        #     t0=k_axis[1],
        #     t1=k_axis[-1],
        #     dt0=1.e-4,     
        #     y0=0.,
        #     stepsize_controller=controller,  # Use the PID step size controller
        #     max_steps = 10000
        # )
        # return 4.*jnp.pi*sol.ys[0]

    def scalar_E_source(self, k, ell, lna_axis, Fg2_func, Gg0_func, Gg2_func, BG):
        """
        Computes the scalar source function for E-mode polarization. 
        See Eq.(3.2) of https://iopscience.iop.org/article/10.1088/1475-7516/2014/09/032/pdf

        Params:
            k : float
            ell : float
            lna_axis : jnp.array
            Fg2 : jnp.array (same shape as lna_axis)
            Gg0 : jnp.array (same shape as lna_axis)
            Gg2 : jnp.array (same shape as lna_axis)
            BG  : cosmology.Background
        """
        N = jnp.sqrt(6.)/8.*jnp.sqrt(3./8. * (ell+2.) * (ell+1.) * ell * (ell-1.))
        tau0 = BG.tau0
        
        def integrand_func(lna, y, args):
            tau = BG.tau(lna)
            g   = BG.visibility(lna)
            aH  = BG.aH(lna)
            chi = k*(tau0-tau)
            Fg2 = Fg2_func(jnp.array([k, lna]))
            Gg0 = Fg2_func(jnp.array([k, lna]))
            Gg2 = Fg2_func(jnp.array([k, lna]))
            integrand  = g/aH * (Fg2+Gg0+Gg2) * jl(ell, chi, 1.e-14)[0] #/ chi**2
            return integrand[0]

        def stop_func(state, **kwargs):
            """
            Stops the solver when tau becomes late enough that we approach the x=0 part of the Bessel function.
            This part of the integrand is zero, and stopping before avoids a numerical divergence.
            """
            lna = state.tprev
            tau = BG.tau(lna)
            chi = k*(tau0-tau)
            return chi < ell/2.

        term = ODETerm(integrand_func)
        controller = PIDController(rtol=1.e-2, atol=1.e-2)

        sol = diffeqsolve(
            term,
            solver=Kvaerno5(),
            t0=lna_axis[0],
            t1=lna_axis[-1],
            dt0=1.e-4,     
            y0=0.,
            stepsize_controller=controller,  # Use the PID step size controller
            max_steps = 10000,
            discrete_terminating_event = DiscreteTerminatingEvent(stop_func)
        )
        return N*sol.ys[0]
