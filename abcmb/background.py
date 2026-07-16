import os

import diffrax
import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from diffrax import (
    ForwardMode,
    Kvaerno5,
    ODETerm,
    PIDController,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from jax import config, lax, vmap

from . import ABCMBTools as tools
from . import constants as cnst
from .hyrex import recomb_functions
from .hyrex.array_with_padding import array_with_padding
from .hyrex.hyrex import RecombInputs

file_dir = os.path.dirname(__file__)
config.update("jax_enable_x64", True)


class BackgroundPreRecomb(eqx.Module):
    """
    Pre-recombination background-cosmology object.

    Holds everything HyRex needs to run on CPU: (conformal-time tabulation,
    species list, and HyRex input arrays via ``RecombInputs`` object).

    Attributes:
    -----------
    species_list : tuple
        A list of all fluids in the cosmology
    lna_tau_tab : jnp.array
        Log scale factor axis used to tabulate conformal time
    tau_tab : jnp.array
        Tabulated conformal time.
    tau0 : float
        Conformal time today in Mpc.
    recomb_inputs : RecombInputs
        Bundle of background quantities (TCMB, nH, H) sampled on
        ``RecModel.lna_axis_full``; consumed by HyRex.
    adjoint : diffrax.adjoint
        Adjoint mode for diffrax solves (static field).

    Methods:
    --------
    rho_tot : Compute total energy density (units: eV cm^{-3})
    P_tot : Compute total pressure (units: eV cm^{-3})
    H : Compute Hubble parameter (units: s^{-1})
    aH : Compute conformal Hubble parameter (units: Mpc^{-1})
    aH_prime : Compute derivative of conformal Hubble (units: Mpc^{-1})
    d2adtau2_over_a : Compute second derivative of scale factor (units: Mpc^{-2})
    tau : Compute conformal time (units: Mpc)
    nH : Compute hydrogen number density (units: cm^{-3})
    TCMB : Compute CMB temperature (units: eV)
    R_ratio_lna : Compute baryon drag ratio (units: dimensionless)
    """

    species_list: tuple

    lna_tau_tab = jnp.linspace(-33.0, 0.0, 10000)  # Axis for tabulating conformal time.
    tau_tab: jnp.array  # Tabulated conformal time.
    tau0: float  # Conformal time today

    recomb_inputs: "RecombInputs"

    adjoint: "diffrax.adjoint" = eqx.field(static=True)

    def __init__(self, params, species_list, RecModel, adjoint=ForwardMode):
        """
        Initialize pre-recombination background.

        Tabulates conformal time and builds the RecombInputs object for
        HyRex.

        Parameters:
        -----------
        params : dict
            Cosmological parameters
        species_list : tuple
            List of fluid species for energy density calculations
        RecModel : hyrex.recomb_model
            Recombination module for computing xe and Tm histories
        adjoint : diffrax.adjoint, optional
            Adjoint class for diffrax solves (default: ForwardMode)
        """
        self.adjoint = adjoint
        self.species_list = species_list

        self.tau_tab = self._tabulate_conformal_time(params)
        self.tau0 = self.tau(0.0)

        # Bundle the background quantities HyRex needs onto its sampling
        # grid (acccording to the input RecModel)
        lna_axis = RecModel.lna_axis_full
        self.recomb_inputs = RecombInputs(
            lna_grid=lna_axis,
            TCMB_arr=vmap(self.TCMB, in_axes=[0, None])(lna_axis, params),
            nH_arr=vmap(self.nH, in_axes=[0, None])(lna_axis, params),
            H_arr=vmap(self.H, in_axes=[0, None])(lna_axis, params),
        )

    def rho_tot(self, lna, params):
        """
        Compute total energy density.

        Sums energy density over all species in the universe.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            Total energy density (units: eV cm^{-3})
        """
        rho_tot = 0.0
        for i in range(len(self.species_list)):
            rho_tot += self.species_list[i].rho(lna, params)
        return rho_tot

    def P_tot(self, lna, params):
        """
        Compute total pressure.

        Sums pressure over all species in the universe.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            Total pressure (units: eV cm^{-3})
        """
        P_tot = 0.0
        for i in range(len(self.species_list)):
            P_tot += self.species_list[i].P(lna, params)
        return P_tot

    def H(self, lna, params):
        """
        Compute Hubble parameter.

        Uses Einstein equation H = sqrt(8πG/3 ρ_tot) to account for
        novel species without well-defined density parameters.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            Hubble parameter (units: s^{-1})
        """
        return jnp.sqrt(8.0 * jnp.pi * cnst.G * self.rho_tot(lna, params) / 3.0)

    def aH(self, lna, params):
        """
        Compute conformal Hubble parameter.

        Calculates conformal Hubble H = a*H = da/dτ where τ is conformal time.
        Uses Mpc units for perturbation calculations.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            Conformal Hubble parameter (units: Mpc^{-1})
        """
        return jnp.exp(lna) * self.H(lna, params) / cnst.c_Mpc_over_s

    def aH_prime(self, lna, params):
        """
        Compute derivative of conformal Hubble parameter.

        Uses second Friedmann equation to compute d(aH)/d(ln a).
        See Eq.(20) of arXiv:9506072.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            Derivative of conformal Hubble (units: Mpc^{-1})
        """
        return (
            -4.0
            * jnp.pi
            * cnst.G
            * jnp.exp(lna) ** 2
            / 3.0
            / self.aH(lna, params)
            * (self.rho_tot(lna, params) + 3.0 * self.P_tot(lna, params))
            / cnst.c_Mpc_over_s**2
        )

    def d2adtau2_over_a(self, lna, params):
        """
        Compute second derivative of scale factor.

        Calculates d²a/dτ²/a where τ is conformal time.
        Appears in perturbation evolution equations.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            Second derivative of scale factor (units: Mpc^{-2})
        """
        return self.aH(lna, params) ** 2 + self.aH(lna, params) * self.aH_prime(
            lna, params
        )

    def _dtau_dlna(self, lna, y, args):
        """
        Compute derivative of conformal time with respect to ln(a).

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : float
            Current conformal time value
        args : dict
            Cosmological parameters

        Returns:
        --------
        float
            Derivative dτ/d(ln a) (units: Mpc)
        """
        params = args
        return 1.0 / self.aH(lna, params)

    def _tabulate_conformal_time(self, params):
        """
        Tabulate conformal time as function of ln(a).

        Integrates dτ/d(ln a) = 1/aH from early times to today
        using radiation-dominated initial conditions. We stitch an
        analytic early-time solution to a Diffrax dense
        interpolation, taking care not to evaluate out of bounds.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        array
            Tabulated conformal time values (units: Mpc)
        """

        lna_cut = -16.1  # use analytic approx before this

        # Analytic early-time approximation
        def tau_approx(lna):
            return (
                jnp.exp(lna)
                / (cnst.H0_over_h / cnst.c_Mpc_over_s)
                / jnp.sqrt(params["omega_r"])
            )

        lna_end = self.lna_tau_tab[-1]

        # ---- Diffrax solve (dense interpolation) ----
        term = ODETerm(self._dtau_dlna)
        controller = PIDController(rtol=1e-8, atol=1e-8)
        saveat = SaveAt(dense=True)
        adjoint = self.adjoint()

        sol = diffeqsolve(
            term,
            solver=Kvaerno5(),
            t0=lna_cut,
            t1=lna_end,
            dt0=1e-5,
            y0=tau_approx(lna_cut),
            saveat=saveat,
            stepsize_controller=controller,
            args=params,
            adjoint=adjoint,
        )

        # Numerical jitter causes this interpolation to go out of bounds on
        # some machines, so we do some extra work to safeguard that here:

        # Strictly inside [lna_cut, lna_end); avoid touching internal sol.ts (may be None).
        # nextafter gets the next representable float below lna_end to ensure in-bounds.
        lna_hi = jnp.nextafter(lna_end, -jnp.inf)

        def _tau_from_sol(l):
            l_in = jnp.clip(l, lna_cut, lna_hi)
            return sol.evaluate(l_in)

        def _tau_combined(l):
            # cond is faster than where since untaken branch is not evaluated
            return lax.cond(l > lna_cut, _tau_from_sol, tau_approx, l)

        tau_tab = vmap(_tau_combined)(self.lna_tau_tab)

        # Replace any remaining non-finite entries with analytic fallback
        tau_tab = jnp.where(
            jnp.isfinite(tau_tab), tau_tab, vmap(tau_approx)(self.lna_tau_tab)
        )

        return tau_tab

    def tau(self, lna):
        """
        Compute conformal time.

        Interpolates from pre-tabulated conformal time history.
        Conformal time τ satisfies dτ = dt/a where t is cosmic time.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            Conformal time (units: Mpc)

        Notes:
        ------
        IDEA: Make Background a repeatedly initiated module with both
        species_list and params stored. Upon initiation, a full history
        of conformal time is calculated with diffrax and stored for
        interpolation. This can be done by approximating early time with
        radiation approximation, and starting diffrax integration at the
        early time with appropriate initial conditions.
        """
        return tools.fast_interp(
            lna, self.lna_tau_tab[0], self.lna_tau_tab[-1], self.tau_tab
        )

    def nH(self, lna, params):
        """
        Compute hydrogen number density.

        Calculates total hydrogen number density at given redshift.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            Hydrogen number density (units: cm^{-3})
        """
        return (
            (1 - params["YHe"])
            * 3.0
            * params["omega_b"]
            * cnst.H0_over_h**2
            / 8
            / jnp.pi
            / cnst.G
            / cnst.mH
            / jnp.exp(lna) ** 3
        )

    def TCMB(self, lna, params):
        """
        Compute CMB temperature.

        Calculates CMB temperature at given redshift using T ∝ 1/a scaling.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            CMB temperature (units: eV)
        """
        return params["TCMB0"] / jnp.exp(lna)

    def R_ratio_lna(self, lna, params):
        """
        Compute baryon drag ratio.

        Calculates R = 3ρ_b/(4ρ_γ), the ratio of baryon to photon
        energy densities that appears in baryon drag calculations.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            Baryon drag ratio (units: dimensionless)
        """
        rho_b = 0.0
        rho_g = 0.0

        for s in self.species_list:
            if s.name == "Photon":
                rho_g += s.rho(lna, params)
            elif s.name == "Baryon":
                rho_b += s.rho(lna, params)

        return 3.0 * rho_b / (4 * rho_g)


class Background(BackgroundPreRecomb):
    """
    Full Background cosmology module for cosmological calculations.

    Inherits all cosmology fields and methods from ``BackgroundPreRecomb``.
    Construction takes a ``BackgroundPreRecomb`` and the recombination output
    from HyRex, then applies reionization and integrates the optical depth.

    This factorization allows HyRex to always run on CPU (its faster backend).

    Attributes:
    -----------
    species_list : tuple
        A list of all fluids in the cosmology
    lna_tau_tab : jnp.array
        Log scale factor axis used to tabulate conformal time
    tau_tab : jnp.array
        Tabulated conformal time.
    tau0 : float
        Conformal time today in Mpc.
    recomb_inputs : RecombInputs
        Bundle of background quantities (TCMB, nH, H) sampled on
        ``RecModel.lna_axis_full``; consumed by HyRex.
    adjoint : diffrax.adjoint
        Adjoint mode for diffrax solves (static field).
    xe_tab : array_with_padding
        Tabulated free electron fraction xe with reionization correction.
    lna_xe_tab : array_with_padding
        Log scale factor axis corresponding to tabulated xe values.
    Tm_tab : array_with_padding
        Tabulated matter temperature Tm during recombination.
    lna_Tm_tab : array_with_padding
        Log scale factor axis corresponding to tabulated Tm values.
    kappa_func : diffrax.solution
        Optical depth function (dense interpolation).
    z_reion : float
        Redshift of hydrogen reionization in the CAMB parameterization.
    tau_reion : float
        Optical depth to reionization.
    lna_rec : float
        Log scale factor of recombination.
    rA_rec : float
        Comoving angular diameter distance at recombination in Mpc.
    lna_transfer_start : float
        Log scale factor at which to begin integrating transfer functions.
    lna_visibility_stop : float
        Log scale factor at which to stop integrating T1, T2, and E sources
        due to small visibility functions. Only used for l<400.

    Methods:
    --------
    rho_tot : Compute total energy density (units: eV cm^{-3})
    P_tot : Compute total pressure (units: eV cm^{-3})
    H : Compute Hubble parameter (units: s^{-1})
    aH : Compute conformal Hubble parameter (units: Mpc^{-1})
    aH_prime : Compute derivative of conformal Hubble (units: Mpc^{-1})
    d2adtau2_over_a : Compute second derivative of scale factor (units: Mpc^{-2})
    tau : Compute conformal time (units: Mpc)
    xe : Compute free electron fraction (units: dimensionless)
    Tm : Compute matter temperature (units: eV)
    tau_c : Compute Thomson scattering time (units: Mpc)
    expmkappa : Compute exp(-kappa) (units: dimensionless)
    visibility : Compute visibility function (units: Mpc^{-1})
    z_d : Compute baryon decoupling redshift (units: dimensionless)
    rs_d : Compute sound horizon at decoupling (units: Mpc)
    """

    xe_tab: "array_with_padding"
    lna_xe_tab: "array_with_padding"
    Tm_tab: "array_with_padding"
    lna_Tm_tab: "array_with_padding"
    kappa_func: "diffrax.solution"
    z_reion: float
    tau_reion: float
    lna_rec: float
    rA_rec: float  # Comoving angular diameter distance at recombination.

    # Transfer related
    lna_transfer_start: float  # Time where transfer functions start integrating.
    lna_visibility_stop: float  # Time to stop integrating T1, T2, and E sources due to small visibility functions. Only used for l<400

    def __init__(self, pre_BG, recomb_output, params, ReionModel):
        """
        Initialize Background cosmology module.

        Consolidates pre-recombination and recombination elements of background cosmology.

        Parameters:
        -----------
        pre_BG : BackgroundPreRecomb
            Output of the pre-recomb stage; provides species_list,
            tau_tab, tau0, recomb_inputs, adjoint.
        recomb_output : tuple
            HyRex output ``(xe, lna_xe, Tm, lna_Tm)`` quadruple
        params : dict
            Cosmological parameters.
        ReionModel : callable
            Reionization module for computing the xe correction.
        """
        # Copy pre-recomb fields onto self.
        self.adjoint = pre_BG.adjoint
        self.species_list = pre_BG.species_list
        self.tau_tab = pre_BG.tau_tab
        self.tau0 = pre_BG.tau0
        self.recomb_inputs = pre_BG.recomb_inputs

        # Unpack HyRex output and apply reionization.
        xe, self.lna_xe_tab, self.Tm_tab, self.lna_Tm_tab = recomb_output

        reion_model = ReionModel(self, params)
        self.z_reion = reion_model.z_reion
        self.tau_reion = reion_model.tau_reion

        xe_reion_correction = reion_model.xe_reion(
            self.lna_xe_tab.arr, self.z_reion, params
        )
        xe_full_arr = xe_reion_correction + xe.arr
        self.xe_tab = array_with_padding(xe_full_arr)

        # Replace inf padding in the recomb tabs with `lastval`. Forward
        # selects the same branch either way (the `where` in BG.xe/BG.Tm
        # gates the fast_interp dead branch out for lna in range). The inf
        # otherwise poisons the lensing=True reverse-AD cotangent: under
        # Kvaerno5+VeryChord, the IFT replay materializes the stage Jacobian
        # via vmap(jvp(RHS)) and chains a cotangent through the where's dead
        # branch into fast_interp past `lastnum`, giving 0×inf = NaN.
        def _finite_pad(awp):
            finite_arr = jnp.where(jnp.isinf(awp.arr), awp.lastval, awp.arr)
            return eqx.tree_at(lambda t: t.arr, awp, finite_arr)

        self.xe_tab = _finite_pad(self.xe_tab)
        self.lna_xe_tab = _finite_pad(self.lna_xe_tab)
        self.Tm_tab = _finite_pad(self.Tm_tab)
        self.lna_Tm_tab = _finite_pad(self.lna_Tm_tab)

        self.kappa_func = self._tabulate_optical_depth(params)

        # Find approximate maximum of visibility function.
        lna_vals = jnp.linspace(-8.0, -4.0, 1500)  # Decoupling falls in here.
        vis_vals = vmap(self.visibility, in_axes=[0, None])(lna_vals, params)
        self.lna_rec = lna_vals[jnp.argmax(vis_vals)]
        self.lna_visibility_stop = lna_vals[jnp.argmin((vis_vals - 1.0e-3) ** 2)]
        self.rA_rec = self.tau0 - self.tau(self.lna_rec)

        # Find approximate early time when aH x tau_c = 0.008
        lna_vals = jnp.linspace(-15.0, -6.0, 5000)
        aH_tau_c_vals = vmap(self.aH, in_axes=[0, None])(lna_vals, params) * self.tau_c(
            lna_vals, params
        )
        self.lna_transfer_start = lna_vals[jnp.argmin((aH_tau_c_vals - 0.008) ** 2)]

    ### RECOMBINATION RELATED ###

    def xe(self, lna):
        """
        Compute free electron fraction.

        Interpolates from pre-tabulated recombination history with
        boundary conditions for early and late times.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            Free electron fraction (units: dimensionless)

        Notes:
        ------
        The logic flow is equivalent to:

        if lna < self.lna_xe_tab.arr[0]: return self.xe_tab[0]
        elif lna > self.lna_xe_tab.lastval: return self.xe_tab.lastval
        else: return jnp.interp(lna, self.lna_xe_tab, self.xe_tab)
        """
        return jnp.where(
            lna < self.lna_xe_tab.arr[0],
            self.xe_tab.arr[0],
            jnp.where(
                lna >= self.lna_xe_tab.lastval,
                self.xe_tab.lastval,
                tools.fast_interp(
                    lna,
                    self.lna_xe_tab.arr[0],
                    self.lna_xe_tab.arr[0]
                    + len(self.lna_xe_tab.arr)
                    * (self.lna_xe_tab.arr[1] - self.lna_xe_tab.arr[0]),
                    self.xe_tab.arr,
                ),
            ),
        )

    def _Tm_early_approx(self, lna, params):
        """
        Compute matter temperature using post-equilibrium approximation.

        Uses approximation Tm = TCMB * (1 - H/GammaCompton) for early times
        before detailed recombination calculation begins.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            Matter temperature (units: eV)
        """
        TCMB = self.TCMB(lna, params)
        xe = self.xe(lna)
        return TCMB * (
            1.0
            - self.H(lna, params)
            / recomb_functions.Gamma_compton(xe, TCMB, params["YHe"])
        )

    def Tm(self, lna, params):
        """
        Compute matter temperature.

        Interpolates from pre-tabulated recombination history with
        early-time approximation and late-time boundary conditions.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            Matter temperature (units: eV)
        """
        return jnp.where(
            lna < self.lna_Tm_tab.arr[0],
            self._Tm_early_approx(lna, params),
            jnp.where(
                lna >= self.lna_Tm_tab.lastval,
                self.Tm_tab.lastval,
                tools.fast_interp(
                    lna,
                    self.lna_Tm_tab.arr[0],
                    self.lna_Tm_tab.arr[0]
                    + len(self.lna_Tm_tab.arr)
                    * (self.lna_Tm_tab.arr[1] - self.lna_Tm_tab.arr[0]),
                    self.Tm_tab.arr,
                ),
            ),
        )

    def tau_c(self, lna, params):
        """
        Compute Thomson scattering time.

        Calculates Thomson scattering time scale τc = 1/(a × ne × σT).

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            Thomson scattering time (units: Mpc)
        """
        a = jnp.exp(lna)
        nH = self.nH(lna, params)
        ne = nH * self.xe(lna)
        return 1.0 / a / ne / cnst.thomson_xsec / cnst.c * cnst.c_Mpc_over_s

    def _tabulate_optical_depth(self, params):
        """
        Tabulate optical depth from given scale factor to today.

        Integrates dκ/d(ln a) = -1/(τc × aH) backwards from today
        to compute optical depth κ(a) = ∫[a to 1] dκ/da' da'.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        array
            Tabulated optical depth values (units: dimensionless)

        Notes:
        ------
        Also computes time derivative of optical depth, which is the
        integrand involving the free electron fraction.
        """

        def integrand(lna, y, args):
            return -1.0 / self.tau_c(lna, params) / self.aH(lna, params)

        term = ODETerm(integrand)
        stepsize_controller = PIDController(
            pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.0e-10, atol=1.0e-10
        )
        adjoint = self.adjoint()
        sol = diffeqsolve(
            term,
            solver=Kvaerno5(),
            stepsize_controller=stepsize_controller,
            t0=0.0,
            t1=-10.0,
            dt0=-1.0e-3,
            max_steps=2048,
            y0=0.0,
            saveat=SaveAt(dense=True),
            adjoint=adjoint,
        )
        return sol

    def expmkappa(self, lna):
        """
        Compute exp(-optical depth).

        Interpolates from pre-tabulated optical depth history.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            exp(-(optical depth)) (units: dimensionless)
        """
        return jnp.where(lna < -10.0, 0.0, jnp.exp(-self.kappa_func.evaluate(lna)))

    def visibility(self, lna, params):
        """
        Compute visibility function.

        Calculates visibility function g(x) = -aH(x) × κ'(x) × exp(-κ(x))
        where ' = d/dx and x = ln a. Represents probability that a CMB
        photon observed today was last scattered at time x.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            Visibility function (units: Mpc^{-1})

        Notes:
        ------
        Used in computing source functions for CMB anisotropies.
        """
        return self.expmkappa(lna) / self.tau_c(lna, params)

    ###########################################
    ### tools for computing decoupling time ###
    ###########################################

    def find_z_at_kappad_equals_one(self, z, kappa_d):
        """
        Find redshift where baryon optical depth equals unity.

        Interpolates to find z_d such that κ_d(z_d) = 1, marking
        the approximate time of baryon decoupling.

        Parameters:
        -----------
        z : array
            Redshift array
        kappa_d : array
            Baryon optical depth array

        Returns:
        --------
        float
            Decoupling redshift (units: dimensionless)
        """
        # ensure sorted ascending
        idx = jnp.argsort(z)
        z_sorted = z[idx]
        kappa_d_sorted = jnp.abs(kappa_d)[idx]

        z_d = jnp.interp(1.0, kappa_d_sorted, z_sorted)
        return z_d

    def interp_rs_at_z(self, z_bg, r_s, z_d):
        """
        Interpolate sound horizon at decoupling redshift.

        Parameters:
        -----------
        z_bg : array
            Background redshift array
        r_s : array
            Sound horizon array
        z_d : float
            Decoupling redshift

        Returns:
        --------
        float
            Sound horizon at decoupling (units: Mpc)
        """
        idx = jnp.argsort(z_bg)
        z_sorted = z_bg[idx]
        rs_sorted = r_s[idx]
        return jnp.interp(z_d, z_sorted, rs_sorted)

    def _tabulate_kappa_d(self, params):
        """
        Tabulate baryon optical depth.

        Integrates dκ_d/d(ln a) = -1/(τc × aH × R) backwards from today
        to compute baryon optical depth including drag effects.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        array
            Tabulated baryon optical depth values (units: dimensionless)
        """

        def integrand(lna, y, args):
            return jnp.float64(
                -1.0
                / self.tau_c(lna, params)
                / self.aH(lna, params)
                / (self.R_ratio_lna(lna, params))
            )

        term = ODETerm(integrand)
        stepsize_controller = PIDController(
            pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.0e-3, atol=1.0e-6
        )
        adjoint = self.adjoint()

        solution = diffeqsolve(
            term,
            solver=Tsit5(),  # Kvaerno5 is just slower but gives same result
            stepsize_controller=stepsize_controller,
            t0=self.lna_tau_tab[-1],  # Initial x value (~0 in this case)
            t1=self.lna_tau_tab[0],  # Final x value (smallest x value)
            dt0=-1e-3,
            max_steps=2048,
            y0=0.0,  # Initial value tau(x=0) = 0
            saveat=SaveAt(
                ts=self.lna_tau_tab[::-1]
            ),  # Save at all points in x, reverse order since integrating backwards
            adjoint=adjoint,
        )
        result = solution.ys[::-1]
        return result

    def _tabulate_rs(self, params):
        """
        Tabulate sound horizon evolution.

        Integrates drs/d(ln a) = cs/aH from early times to today
        where cs = 1/√(3(1+R)) accounts for baryon loading.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        array
            Tabulated sound horizon values (units: Mpc)
        """
        # initial condition assuming cs**2 = 1/3 at early times
        rs0 = 1.0 / jnp.sqrt(3) / (self.aH(self.lna_tau_tab[0], params))

        def integrand(lna, y, args):
            return (
                1.0
                / jnp.sqrt(3 * (1 + self.R_ratio_lna(lna, params)))
                / (self.aH(lna, params))
            )

        term = ODETerm(integrand)
        stepsize_controller = PIDController(
            pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.0e-3, atol=1.0e-6
        )
        adjoint = self.adjoint()

        solution = diffeqsolve(
            term,
            solver=Tsit5(),
            stepsize_controller=stepsize_controller,
            t0=self.lna_tau_tab[0],  # reversed direction since I know rs at early times
            t1=self.lna_tau_tab[-1],
            dt0=1e-3,
            max_steps=2048,
            y0=rs0,
            saveat=SaveAt(ts=self.lna_tau_tab),
            adjoint=adjoint,
        )
        result = solution.ys
        return result

    def z_d(self, params):
        """
        Compute baryon decoupling redshift.

        Finds redshift where κ_d = 1 as estimate of when baryons
        decouple from photons.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            Decoupling redshift (units: dimensionless)
        """
        return self.find_z_at_kappad_equals_one(
            1 / jnp.exp(self.lna_tau_tab) - 1, self._tabulate_kappa_d(params)
        )

    def rs_d(self, params):
        """
        Compute sound horizon at decoupling.

        Finds value of sound horizon at baryon decoupling redshift z_d.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            Sound horizon at decoupling (units: Mpc)
        """
        return self.interp_rs_at_z(
            1 / jnp.exp(self.lna_tau_tab) - 1,
            self._tabulate_rs(params),
            self.z_d(params),
        )


class ReionizationModel(eqx.Module):
    """
    Object for computing the reionization correction to the free electron fraction.
    Provides the base methods

    xe_reion : calculates the tanh electron fraction correction at redshifts lna, given z_reion and params
    tau_reion_fn : calculates the optical depth to reionization.

    At the moment we only support the CAMB tanh parameterization, but we need different approaches
    based on whether the use inputs the optical depth tau_reion or the reionization redshift z_reion.

    """

    z_reion: jnp.float64
    tau_reion: jnp.float64

    def xe_reion(self, lna, z_reion, params):
        """
        Passing in an lna array should get you the correct tanh patching based on the
        reionization parameter.
        """
        fHe = params["YHe"] / 4 / (1 - params["YHe"])
        z = 1 / jnp.exp(lna) - 1
        y = (1 + z) ** (params["exp_reion"])

        y_reion = (1 + z_reion) ** (params["exp_reion"])
        Delta_y_reion = (
            params["exp_reion"]
            * (1 + z_reion) ** (params["exp_reion"] - 1)
            * params["Delta_z_reion"]
        )
        tanh_arg = (y_reion - y) / Delta_y_reion
        xe_reion_H = (1 + fHe) / 2 * (1 + jnp.tanh(tanh_arg))

        # The above accounts for hydrogen and the first ionization level of helium.
        # Let's also account for the second ionization of helium:
        tanh_arg_He = (params["z_reion_He"] - z) / params["Delta_z_reion_He"]
        xe_reion_HeII = fHe / 2 * (1 + jnp.tanh(tanh_arg_He))

        return xe_reion_H + xe_reion_HeII

    def tau_reion_fn(self, z_reion, BG, params):
        lna_axis = jnp.linspace(-5.0, 0.0, 2000)
        xe_reion_correction = self.xe_reion(lna_axis, z_reion, params)
        # Free electron number density belonging only to reionized hydrogen.
        ne = BG.nH(lna_axis, params) * xe_reion_correction
        Gamma = jnp.exp(lna_axis) * ne * cnst.thomson_xsec * cnst.c / cnst.c_Mpc_over_s
        aH = BG.aH(lna_axis, params)
        # Optical depth integrand
        integrand = Gamma / aH
        return jnp.trapezoid(integrand, lna_axis)


class ReionizationModelFromZ(ReionizationModel):
    """
    Concrete extension of the base ReionizationModel Class.
    This object is used when the user direcly inputs the redshift of reionization.
    In this case the tanh correction and the optical depth can be computed directly,
    and simply returned.
    """

    def __init__(self, BG, params):
        self.z_reion = params.get("z_reion", jnp.array(7.6711))
        self.tau_reion = self.tau_reion_fn(self.z_reion, BG, params)


class ReionizationModelFromTau(ReionizationModel):
    """
    Concrete extension of the base ReionizationModel Class.
    This object is used when the user inputs the optical depth and wishes to infer the redshift.
    The init finder will use an optimistix root finder to find the appropriate redshift.
    Then the appropriate tanh correction may be called and returned, as well as the inferred reionization redshift.
    """

    def __init__(self, BG, params):

        def tau_target_fn(z_reion, args):
            target = args
            return self.tau_reion_fn(z_reion, BG, params) - target

        solver = optx.Newton(rtol=1e-5, atol=1e-5)
        sol = optx.root_find(
            tau_target_fn, solver, 7.6, params.get("tau_reion", jnp.array(0.05430842))
        )
        self.z_reion = sol.value
        self.tau_reion = params.get("tau_reion", jnp.array(0.05430842))
