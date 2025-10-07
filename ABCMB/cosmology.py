import jax
from jax import config, vmap, lax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, Kvaerno5, Tsit5, SaveAt, PIDController, ForwardMode

import ABCMB.AbstractSpecies as AS
from .hyrex.array_with_padding import array_with_padding
from .hyrex import recomb_functions
from . import ABCMBTools as tools
from . import constants as cnst

import os
file_dir = os.path.dirname(__file__)
config.update("jax_enable_x64", True)


class Background(eqx.Module):
    """
    Background cosmology module for cosmological calculations.

    Computes background quantities including Hubble parameter, conformal time,
    recombination history, and optical depth evolution.

    Recombination Unrelated Methods:
    --------------------------------
    rho_tot : Compute total energy density (units: eV cm^{-3})
    P_tot : Compute total pressure (units: eV cm^{-3})
    H : Compute Hubble parameter (units: s^{-1})
    aH : Compute conformal Hubble parameter (units: Mpc^{-1})
    aH_prime : Compute derivative of conformal Hubble (units: Mpc^{-1})
    d2adtau2_over_a : Compute second derivative of scale factor (units: Mpc^{-2})
    tau : Compute conformal time (units: Mpc)
    z_d : Compute baryon decoupling redshift (units: dimensionless)
    rs_d : Compute sound horizon at decoupling (units: Mpc)

    Recombination Related Methods:
    ------------------------------
    xe : Compute free electron fraction (units: dimensionless)
    Tm : Compute matter temperature (units: eV)
    mu_bar : Compute mean molecular mass (units: eV)
    cs2 : Compute baryon sound speed squared (units: dimensionless)
    nH : Compute hydrogen number density (units: cm^{-3})
    TCMB : Compute CMB temperature (units: eV)
    tau_c : Compute Thomson scattering time (units: Mpc)
    kappa : Compute optical depth (units: dimensionless)
    visibility : Compute visibility function (units: Mpc^{-1})
    """

    params : dict
    species_list : tuple
 
    lna_tau_tab = jnp.linspace(-33.0, 0.0, 10000) # Axis for tabulating conformal time.
    tau_tab : jnp.array                     # Tabulated conformal time. 
    tau0 : float # Conformal time today

    # Recombination related
    xe_tab     : "array_with_padding"
    lna_xe_tab : "array_with_padding"
    Tm_tab     : "array_with_padding"
    lna_Tm_tab : "array_with_padding"
    kappa_tab  : jnp.array
    lna_rec    : float
    rA_rec     : float # Comoving angular diameter distance at recombination.
    rs_d       : float # Sound horizon at baryon decoupling
    z_d        : float # redshift of baryon devoupling

    # Transfer related
    lna_transfer_start : float # Time where transfer functions start integrating.
    lna_visibility_stop : float # Time to stop integrating T1, T2, and E sources due to small visibility functions. Only used for l<400

    def __init__(self, params, species_list, RM):
        """
        Initialize Background cosmology module.

        Computes and tabulates conformal time, recombination history,
        optical depth, and key cosmological epochs.

        Parameters:
        -----------
        params : dict
            Cosmological parameters
        species_list : tuple
            List of fluid species for energy density calculations
        RM : callable
            Recombination module for computing xe and Tm histories
        """
        self.params = params
        self.species_list = species_list

        self.tau_tab = self._tabulate_conformal_time()
        self.tau0 = self.tau(0.)
        
        ### RECOMBINATION RELATED ###

        # Run hyrex to tabulate recombination output
        self.xe_tab, self.lna_xe_tab, self.Tm_tab, self.lna_Tm_tab = RM(self)
        self.kappa_tab = self._tabulate_optical_depth()

        # Find approximate maximum of visibility function.
        lna_vals = jnp.linspace(-8.0, -4.0, 1500) # Decoupling should have happened at some time in this interval.
        vis_vals = self.visibility(lna_vals)
        self.lna_rec = lna_vals[jnp.argmax(vis_vals)]
        self.lna_visibility_stop = lna_vals[jnp.argmin((vis_vals - 1.e-3)**2)]
        self.rA_rec = self.tau0 - self.tau(self.lna_rec)

        # Find approximate early time when aH x tau_c = 0.008
        lna_vals = jnp.linspace(-15.0, -6.0, 5000)
        aH_tau_c_vals = vmap(self.aH)(lna_vals)*self.tau_c(lna_vals)
        self.lna_transfer_start = lna_vals[jnp.argmin((aH_tau_c_vals-0.008)**2)]


    def rho_tot(self, lna):
        """
        Compute total energy density.

        Sums energy density over all species in the universe.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            Total energy density (units: eV cm^{-3})

        Notes:
        ------
        User should not modify this function without careful consideration.
        """
        rho_tot = 0.
        for i in range(len(self.species_list)):
            rho_tot += self.species_list[i].rho(lna, self)
        return rho_tot
    
    def P_tot(self, lna):
        """
        Compute total pressure.

        Sums pressure over all species in the universe.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            Total pressure (units: eV cm^{-3})

        Notes:
        ------
        User should not modify this function without careful consideration.
        """
        P_tot = 0.
        for i in range(len(self.species_list)):
            P_tot += self.species_list[i].P(lna, self)
        return P_tot

    def H(self, lna):
        """
        Compute Hubble parameter.

        Uses Einstein equation H = sqrt(8πG/3 ρ_tot) to account for
        novel species without well-defined density parameters.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            Hubble parameter (units: s^{-1})
        """
        return jnp.sqrt(8.*jnp.pi*cnst.G*self.rho_tot(lna)/3.)

    def aH(self, lna):
        """
        Compute conformal Hubble parameter.

        Calculates conformal Hubble H = a*H = da/dτ where τ is conformal time.
        Uses Mpc units for perturbation calculations.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            Conformal Hubble parameter (units: Mpc^{-1})
        """
        return jnp.exp(lna)*self.H(lna) / cnst.c_Mpc_over_s
    
    def aH_prime(self, lna):
        """
        Compute derivative of conformal Hubble parameter.

        Uses second Friedmann equation to compute d(aH)/d(ln a).
        See Eq.(20) of arXiv:9506072.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            Derivative of conformal Hubble (units: Mpc^{-1})
        """
        return -4.*jnp.pi*cnst.G*jnp.exp(lna)**2/3./self.aH(lna) * (self.rho_tot(lna)+3.*self.P_tot(lna)) / cnst.c_Mpc_over_s**2

    def d2adtau2_over_a(self, lna):
        """
        Compute second derivative of scale factor.

        Calculates d²a/dτ²/a where τ is conformal time.
        Appears in perturbation evolution equations.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            Second derivative of scale factor (units: Mpc^{-2})
        """

        return self.aH(lna)**2 + self.aH(lna)*self.aH_prime(lna)
    
    def _dtau_dlna(self, lna, y, args):
        """
        Compute derivative of conformal time with respect to ln(a).

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : float
            Current conformal time value
        args : tuple
            Additional arguments (unused)

        Returns:
        --------
        float
            Derivative dτ/d(ln a) (units: Mpc)
        """
        return 1./self.aH(lna)

    def _tabulate_conformal_time(self):
        """
        Tabulate conformal time as function of ln(a).

        Integrates dτ/d(ln a) = 1/aH from early times to today
        using radiation-dominated initial conditions. We stitch an 
        analytic early-time solution to a Diffrax dense 
        interpolation, taking care not to evaluate out of bounds.

        Returns:
        --------
        array
            Tabulated conformal time values (units: Mpc)
 
        Tabulate conformal time τ(ln a).
        """

        lna_cut = -16.1  # use analytic approx before this
        # Analytic early-time approximation
        tau_approx = lambda lna: (
            jnp.exp(lna) / (cnst.H0_over_h / cnst.c_Mpc_over_s) / jnp.sqrt(self.params["omega_r"])
        )

        lna_end = self.lna_tau_tab[-1]

        # ---- Diffrax solve (dense interpolation) ----
        term = ODETerm(self._dtau_dlna)
        controller = PIDController(rtol=1e-8, atol=1e-8)
        saveat = SaveAt(dense=True)
        adjoint=ForwardMode()

        sol = diffeqsolve(
            term,
            solver=Kvaerno5(),
            t0=lna_cut,
            t1=lna_end,
            dt0=1e-5,
            y0=tau_approx(lna_cut),
            saveat=saveat,
            stepsize_controller=controller,
            adjoint=adjoint
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
        tau_tab = jnp.where(jnp.isfinite(tau_tab), tau_tab, vmap(tau_approx)(self.lna_tau_tab))

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

        return tools.fast_interp(lna, self.lna_tau_tab[0], self.lna_tau_tab[-1], self.tau_tab)

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
        if lna < self.lna_xe_tab.arr[0]:
            return self.xe_tab[0]
        elif lna > self.lna_xe_tab.lastval
            return self.xe_tab.lastval
        else
            return jnp.interp(lna, self.lna_xe_tab, self.xe_tab)
        """
        return jnp.where(
            lna < self.lna_xe_tab.arr[0],
            self.xe_tab.arr[0],
            jnp.where(
                lna >= self.lna_xe_tab.lastval,
                self.xe_tab.lastval,
                tools.fast_interp(lna, self.lna_xe_tab.arr[0],
                self.lna_xe_tab.arr[0] + len(self.lna_xe_tab.arr) * (self.lna_xe_tab.arr[1]-self.lna_xe_tab.arr[0]),
                self.xe_tab.arr)
            )
        )

    def _Tm_early_approx(self, lna):
        """
        Compute matter temperature using post-equilibrium approximation.

        Uses approximation Tm = TCMB * (1 - H/GammaCompton) for early times
        before detailed recombination calculation begins.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            Matter temperature (units: eV)
        """
        TCMB = self.TCMB(lna)
        xe   = self.xe(lna)
        return TCMB * (1.-self.H(lna)/recomb_functions.Gamma_compton(xe, TCMB, self.params['YHe']))

    def Tm(self, lna):
        """
        Compute matter temperature.

        Interpolates from pre-tabulated recombination history with
        early-time approximation and late-time boundary conditions.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            Matter temperature (units: eV)
        """
        return jnp.where(
            lna < self.lna_Tm_tab.arr[0],
            self._Tm_early_approx(lna),
            jnp.where(
                lna >= self.lna_Tm_tab.lastval,
                self.Tm_tab.lastval,
                tools.fast_interp(lna, self.lna_Tm_tab.arr[0], self.lna_Tm_tab.arr[-1], self.Tm_tab.arr)
            )
        )

    def nH(self, lna):
        """
        Compute hydrogen number density.

        Calculates total hydrogen number density at given redshift.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            Hydrogen number density (units: cm^{-3})
        """
        return (1-self.params['YHe']) * 3. * self.params['omega_b'] * cnst.H0_over_h**2 / 8 / jnp.pi / cnst.G / cnst.mH / jnp.exp(lna)**3

    def TCMB(self,lna):
        """
        Compute CMB temperature.

        Calculates CMB temperature at given redshift using T ∝ 1/a scaling.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            CMB temperature (units: eV)
        """
        return self.params['TCMB0'] / jnp.exp(lna)

    def tau_c(self, lna):
        """
        Compute Thomson scattering time.

        Calculates Thomson scattering time scale τc = 1/(a × ne × σT).

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            Thomson scattering time (units: Mpc)
        """
        a = jnp.exp(lna)
        nH = self.nH(lna)
        ne = nH * self.xe(lna)
        return 1./a/ne/cnst.thomson_xsec/cnst.c*cnst.c_Mpc_over_s

    @jax.named_scope("tabulate optical depth")
    def _tabulate_optical_depth(self):
        """
        Tabulate optical depth from given scale factor to today.

        Integrates dκ/d(ln a) = -1/(τc × aH) backwards from today
        to compute optical depth κ(a) = ∫[a to 1] dκ/da' da'.

        Returns:
        --------
        array
            Tabulated optical depth values (units: dimensionless)

        Notes:
        ------
        Also computes time derivative of optical depth, which is the
        integrand involving the free electron fraction.
        """
        integrand = lambda lna, y, args: -1./self.tau_c(lna)/self.aH(lna)
        term = ODETerm(integrand)
        stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.e-3, atol=1.e-6)
        adjoint=ForwardMode()
        solution = diffeqsolve(
            term,
            solver=Kvaerno5(),            # Higher order integrator for more accuracy
            stepsize_controller=stepsize_controller,
            t0=self.lna_tau_tab[-1],                 # Initial x value (~0 in this case)
            t1=self.lna_tau_tab[0],                  # Final x value (smallest x value)
            dt0=-1e-3,                  # Initial step size
            max_steps=2048,
            y0=0.0,                     # Initial value tau(x=0) = 0
            saveat=SaveAt(ts=self.lna_tau_tab[::-1]), # Save at all points in x, reverse order since integrating backwards
            adjoint=adjoint
        )
        result = solution.ys[::-1]
        return result

    def kappa(self, lna):
        """
        Compute optical depth.

        Interpolates from pre-tabulated optical depth history.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            Optical depth (units: dimensionless)
        """
        return tools.fast_interp(lna, self.lna_tau_tab[0], self.lna_tau_tab[-1], self.kappa_tab)

    def visibility(self, lna):
        """
        Compute visibility function.

        Calculates visibility function g(x) = -aH(x) × κ'(x) × exp(-κ(x))
        where ' = d/dx and x = ln a. Represents probability that a CMB
        photon observed today was last scattered at time x.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            Visibility function (units: Mpc^{-1})

        Notes:
        ------
        Used in computing source functions for CMB anisotropies.
        """
        return 1./self.tau_c(lna)*jnp.exp(-self.kappa(lna))

    ###########################################
    ### tools for computing decoupling time ###
    ###########################################

    def find_z_at_kappad_equals_one(self,z, kappa_d):
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

        # interpolate
        z_d = jnp.interp(1.0, kappa_d_sorted, z_sorted)
        return z_d

    def interp_rs_at_z(self,z_bg, r_s, z_d):
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

    def R_ratio_lna(self,lna):
        """
        Compute baryon drag ratio.

        Calculates R = 3ρ_b/(4ρ_γ), the ratio of baryon to photon
        energy densities that appears in baryon drag calculations.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor

        Returns:
        --------
        float
            Baryon drag ratio (units: dimensionless)
        """
        rho_b = self.species_list[3].rho(lna,self)
        rho_g = self.species_list[4].rho(lna,self)
        return 3. * rho_b / (4 * rho_g)

    @jax.named_scope("tabulate kappa d")
    def _tabulate_kappa_d(self):
        """
        Tabulate baryon optical depth.

        Integrates dκ_d/d(ln a) = -1/(τc × aH × R) backwards from today
        to compute baryon optical depth including drag effects.

        Returns:
        --------
        array
            Tabulated baryon optical depth values (units: dimensionless)
        """
        integrand = lambda lna, y, args: jnp.float64(-1./self.tau_c(lna)/self.aH(lna)/(self.R_ratio_lna(lna)))
        term = ODETerm(integrand)
        stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.e-3, atol=1.e-6)
        adjoint=ForwardMode()
        
        solution = diffeqsolve(
            term,
            solver=Tsit5(),            # Kvaerno5 is just slower but gives same result
            stepsize_controller=stepsize_controller,
            t0=self.lna_tau_tab[-1],                 # Initial x value (~0 in this case)
            t1=self.lna_tau_tab[0],                  # Final x value (smallest x value)
            dt0=-1e-3,                  # Initial step size
            max_steps=2048,
            y0=0.0,                     # Initial value tau(x=0) = 0
            saveat=SaveAt(ts=self.lna_tau_tab[::-1]), # Save at all points in x, reverse order since integrating backwards
            adjoint=adjoint
        )
        result = solution.ys[::-1]
        return result

    @jax.named_scope("tabulate rs")
    def _tabulate_rs(self):
        """
        Tabulate sound horizon evolution.

        Integrates drs/d(ln a) = cs/aH from early times to today
        where cs = 1/√(3(1+R)) accounts for baryon loading.

        Returns:
        --------
        array
            Tabulated sound horizon values (units: Mpc)
        """
         # initial condition assuming cs**2 = 1/3 at early times
        rs0 = 1./jnp.sqrt(3) / (self.aH( self.lna_tau_tab[0] ))

        integrand = lambda lna, y, args: 1./jnp.sqrt(3*(1+self.R_ratio_lna(lna))) / (self.aH( lna ))
        term = ODETerm(integrand)
        stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.e-3, atol=1.e-6)
        adjoint=ForwardMode()
        
        solution = diffeqsolve(
            term,
            solver=Tsit5(),
            stepsize_controller=stepsize_controller,
            t0=self.lna_tau_tab[0],                 # reversed direction since I know rs at early times
            t1=self.lna_tau_tab[-1],
            dt0=1e-3,
            max_steps=2048,
            y0=rs0,
            saveat=SaveAt(ts=self.lna_tau_tab),
            adjoint=adjoint
        )
        result = solution.ys
        return result
        
        rs = get_rs(self)

    def z_d(self):
        """
        Compute baryon decoupling redshift.

        Finds redshift where κ_d = 1 as estimate of when baryons
        decouple from photons.

        Returns:
        --------
        float
            Decoupling redshift (units: dimensionless)
        """
        return self.find_z_at_kappad_equals_one(1/jnp.exp(self.lna_tau_tab) - 1, self._tabulate_kappa_d())

    def rs_d(self):
        """
        Compute sound horizon at decoupling.

        Finds value of sound horizon at baryon decoupling redshift z_d.

        Returns:
        --------
        float
            Sound horizon at decoupling (units: Mpc)
        """
        return self.interp_rs_at_z(1/jnp.exp(self.lna_tau_tab) - 1, self._tabulate_rs(), self.z_d())

class MockBackground(Background):

    H_tab : jnp.array

    def __init__(self):
        self.species_list = (AS.ColdDarkMatter(0), AS.DarkEnergy(), AS.Baryon(0, None), AS.Photon(0, None))
        self.params = {k: float(v) for k, v in np.loadtxt(file_dir+"/../Module_Tests/params.txt", dtype=str)}

        # Other tabulated things
        data = np.load(file_dir+"/../Module_Tests/background.npz")

        self.tau_tab = jnp.array(data["tau_tab"])
        self.tau0 = self.tau_tab[-1]
        self.H_tab = jnp.array(data["H_tab"])

        # hyrec_swift = np.loadtxt("/home/zz1994/packages/HYREC-2/ABCMB_test_FULL.dat")
        # self.lna_xe_tab = array_with_padding(jnp.array(-jnp.log(1.+hyrec_swift[:, 0])))
        # self.xe_tab = array_with_padding(jnp.array(hyrec_swift[:, 1]))
        # self.lna_Tm_tab = array_with_padding(jnp.array(-jnp.log(1.+hyrec_swift[:, 0])))
        # self.Tm_tab = array_with_padding(jnp.array(hyrec_swift[:, 2]*cnst.kB))

        self.lna_xe_tab = array_with_padding(jnp.array(data["lna_xe_tab"]))
        self.xe_tab = array_with_padding(jnp.array(data["xe_tab"]))
        self.lna_Tm_tab = array_with_padding(jnp.array(data["lna_Tm_tab"]))
        self.Tm_tab = array_with_padding(jnp.array(data["Tm_tab"]))
        self.kappa_tab = jnp.array(data["kappa_tab"])
        # self.kappa_tab = self._tabulate_optical_depth()

        self.lna_rec = -6.99666444
        self.lna_transfer_start = -7.27285457
        self.lna_visibility_stop = -6.70847231
        self.rA_rec = 13899.20802848

    def H(self, lna):
        return jnp.interp(lna, self.lna_tau_tab, self.H_tab)

    # def xe(self, lna):
    #     return jnp.interp(lna, self.lna_xe_tab.arr, self.xe_tab.arr)

    # def Tm(self, lna):
    #     return jnp.interp(lna, self.lna_Tm_tab.arr, self.Tm_tab.arr)

class ClassBackground(Background):

    H_tab : jnp.array

    def __init__(self):
        self.species_list = (AS.ColdDarkMatter(0), AS.DarkEnergy(), AS.Baryon(0, None), AS.Photon(0, None))
        self.params = {k: float(v) for k, v in np.loadtxt(file_dir+"/../Module_Tests/params.txt", dtype=str)}

        class_res_dir = "/home/zz1994/packages/class/output/ABCMB_test/noneutrinos00"
        bac = np.loadtxt(class_res_dir+"_background.dat")
        therm = np.loadtxt(class_res_dir+"_thermodynamics.dat")

        #self.lna_tau_tab = jnp.array(np.loadtxt("MockBackgroundTabs/lna_tau.txt"))
        self.tau_tab = jnp.interp(self.lna_tau_tab, -jnp.log(1.+bac[:, 0]), bac[:, 2])
        self.tau0 = self.tau_tab[-1]
        self.H_tab = jnp.interp(self.lna_tau_tab, -jnp.log(1.+bac[:, 0]), bac[:, 3]) * cnst.c_Mpc_over_s

        #thermo_class = np.loadtxt("/home/zz1994/packages/class/output/ABCMB_test"+"/noufarsa00_thermodynamics.dat")
        a = therm[:, 0]
        xe_class = therm[:, 3]
        Tm_class = therm[:, 7]
        self.lna_xe_tab = array_with_padding(jnp.array(jnp.log(jnp.flip(a))))
        self.xe_tab = array_with_padding(jnp.array(jnp.flip(xe_class)))
        self.lna_Tm_tab = array_with_padding(jnp.array(jnp.log(jnp.flip(a))))
        self.Tm_tab = array_with_padding(jnp.array(jnp.flip(Tm_class)*cnst.kB))
        self.kappa_tab = jnp.interp(self.lna_tau_tab, -jnp.log(a), -jnp.log(therm[:, 5]))

        self.lna_rec = -6.99666444
        self.lna_transfer_start = -7.27285457
        self.lna_visibility_stop = -6.70847231
        self.rA_rec = 13899.20802848

    def H(self, lna):
        return jnp.interp(lna, self.lna_tau_tab, self.H_tab)

    def xe(self, lna):
        return jnp.interp(lna, self.lna_xe_tab.arr, self.xe_tab.arr)

    def Tm(self, lna):
        return jnp.interp(lna, self.lna_Tm_tab.arr, self.Tm_tab.arr)