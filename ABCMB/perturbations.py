import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, lax
import diffrax
import equinox as eqx

from . import constants as cnst
from . import ABCMBTools as tools

import os
file_dir = os.path.dirname(__file__)
jax.config.update("jax_enable_x64", True)

"""
Cosmological perturbation evolution module.

Integrates linear perturbation equations for scalar modes across
cosmic time using background cosmology and species interactions.
"""


class PerturbationEvolver(eqx.Module):
    """
    Linear scalar perturbation evolution solver.

    Evolves perturbations for all fluid species using Einstein-Boltzmann
    equations in synchronous gauge.

    Methods:
    --------
    full_evolution : Evolve perturbations for multiple k modes
    evolution_one_k : Evolve perturbations for single k mode
    get_tca_on_off : Determine tight coupling approximation times
    initial_conditions_one_k : Compute initial perturbation conditions
    get_derivatives : Compute perturbation time derivatives
    make_output_table : Create interpolatable perturbation table
    """

    perturbations_list : tuple  #= eqx.static_field()
    BG : "cosmology.Background" #= eqx.static_field()

    def full_evolution(self):
        """
        Evolve perturbations for multiple wavenumber modes.

        Integrates perturbation equations for a range of k modes,
        then interpolates results onto common time grid.

        Returns:
        --------
        PerturbationTable
            Interpolatable table of perturbation evolution

        Notes:
        ------
        Uses logarithmic k spacing from 10^-4 to ~0.5 Mpc^-1 with 100 points.
        Time integration runs from early times to z=1 (lna=-ln(2)).
        """
        #k = jnp.logspace(-4., -0.3, 100, base=10)
        k = jnp.geomspace(1.e-4, 0.4, 300)
        #k = jnp.array([1.e-3, 1.e-2, 1.e-1])
        #k = jnp.array([1.e-3, 1.e-2])
        #k = jnp.array([1.e-1])
        sols = vmap(self.evolution_one_k)(k)

        lna = jnp.linspace(self.BG.lna_transfer_start,  0., 500) # lna_end hardcoded
        # lna = jnp.linspace(-15., 0., 1000)
        #lna = jnp.linspace(-15., -7.0, 500)
        res = vmap(lambda arr: vmap(arr.evaluate)(lna))(sols) # Right now the shape is (Nk, Nlna, Ny)
        res = res.transpose(2, 1, 0) # Transpose so the shape is (Ny, Nlna, Nk), easier for vmapping over in PT
        #return lna, k, res

        PT = self.make_output_table(k, lna, res)
        return PT

    def full_evolution_scan(self):
        """
        Evolve perturbations for multiple wavenumber modes.

        Integrates perturbation equations for a range of k modes,
        then interpolates results onto common time grid.

        Returns:
        --------
        PerturbationTable
            Interpolatable table of perturbation evolution
        """
        lna = jnp.linspace(self.BG.lna_transfer_start, 0., 500)  # lna_end hardcoded
        #lna = jnp.linspace(-14., 0., 1000)
        k = jnp.geomspace(1.e-4, 0.4, 300)
        #k = jnp.array([1.e-3, 1.e-2, 1.e-1])

        def body_fun(_, ki):
            # evolution_one_k returns shape (Nlna, Ny)
            y = self.evolution_one_k_scan(ki, lna)    # (Nlna, Ny)
            return None, y

        _, res = lax.scan(body_fun, None, k)      # res has shape (Nk, Nlna, Ny)
        res = res.transpose(2, 1, 0)              # -> (Ny, Nlna, Nk)

        PT = self.make_output_table(k, lna, res)
        return PT

    def get_tca_on_off(self, k):
        """
        Determine tight coupling approximation time range.

        Finds start and end times for tight coupling between photons and baryons
        by computing when Thomson scattering becomes ineffective relative to
        Hubble and horizon crossing time scales.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})

        Returns:
        --------
        tuple
            (lna_start, lna_end) for tight coupling period

        Notes:
        ------
        Uses thresholds: τc/τh < 0.0015 (start), τh/τk < 0.07 (start),
        τc/τh > 0.015 (end), τc/τk > 0.01 (end).
        """
        # thresholds
        thr1 = 0.0015   # start_small_k_at_tau_c_over_tau_h
        thr2 = 0.07     # start_large_k_at_tau_h_over_tau_k
        thr3 = 0.015    # tight_coupling_trigger_tau_c_over_tau_h
        thr4 = 0.01     # tight_coupling_trigger_tau_c_over_tau_k

        # 1) Starting lna
        lna_start_range = jnp.linspace(-20.0, -10.0, 10000)

        # a) τc/τh  →  f1(lna) = BG.tau_c * BG.aH
        f1 = self.BG.tau_c(lna_start_range) * self.BG.aH(lna_start_range)
        # invert f1(lna) = thr1  →  lna = interp(thr1, f1, lna_range)
        lna1 = jnp.interp(thr1, f1, lna_start_range)    # jnp.interp ends up being 
                                                        # faster than fast_interp through here

        # b) τh/τk  →  f2(lna) = k / BG.aH
        f2 = k / self.BG.aH(lna_start_range)
        # invert f2(lna) = thr2
        lna2 = jnp.interp(thr2, f2, lna_start_range)

        lna_ini = jnp.minimum(lna1, lna2)

        # 2) Ending lna
        lna_end_range = jnp.linspace(-15.0, -6.0, 10000)

        # a) τc/τh  →  f3(lna) = BG.tau_c * BG.aH
        f3 = self.BG.tau_c(lna_end_range) * self.BG.aH(lna_end_range)
        lna3 = jnp.interp(thr3, f3, lna_end_range) 

        # b) τc/τk  →  f4(lna) = k * BG.tau_c
        f4 = k * self.BG.tau_c(lna_end_range)
        lna4 = jnp.interp(thr4, f4, lna_end_range)

        lna_end = jnp.minimum(lna3, lna4)

        return lna_ini, lna_end

    def initial_conditions_one_k(self, k, lna_ini):
        """
        Compute initial conditions for perturbation evolution.

        Sets up initial values for metric and fluid perturbations at early times
        using adiabatic initial conditions.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        lna_ini : float
            Initial logarithm of scale factor

        Returns:
        --------
        array
            Initial perturbation state vector

        Notes:
        ------
        Uses CLASS-style initial conditions with metric perturbations h and η.
        Assumes adiabatic initial conditions with vanishing isocurvature modes.
        """
        ### CLASS Initial Conditions ###
        a = jnp.exp(lna_ini)
        tau_ini = self.BG.tau(lna_ini)
 
        rho_crit = 3*self.BG.params["H0"]**2 / 8. / jnp.pi / cnst.G # Crit density today, eV/cm^3
        rho_m = self.BG.params["omega_m"]/self.BG.params["h"]**2 * rho_crit / a**3
        rho_r = self.BG.params["omega_r"]/self.BG.params["h"]**2 * rho_crit / a**4

        om = a*rho_m/jnp.sqrt(rho_r) * jnp.sqrt(8.*jnp.pi*cnst.G/3.) / cnst.c_Mpc_over_s # In units of 1/Mpc

        metric_eta_ini = (1.-k**2*tau_ini**2/12./(15.+4.*self.BG.params['R_nu'])*(5.+4.*self.BG.params['R_nu'] - (16.*self.BG.params['R_nu']*self.BG.params['R_nu']+280.*self.BG.params['R_nu']+325)/10./(2.*self.BG.params['R_nu']+15.)*tau_ini*om))
        metric_h_ini   = 0.5 * (k * tau_ini)**2 * (1.-om*tau_ini/5.)

        all_fluid_ini = jnp.concatenate([p.y_ini(k, tau_ini, om, self.BG) for p in self.perturbations_list])
        y_ini = jnp.concatenate((jnp.array([metric_h_ini, metric_eta_ini]), all_fluid_ini))
        
        return y_ini

    def get_derivatives(self, lna, y, args):
        """
        Compute time derivatives for perturbation evolution.

        Assembles the full system of Einstein-Boltzmann equations for
        metric and fluid perturbations in synchronous gauge.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Current perturbation state vector
        args : float
            Wavenumber k (units: Mpc^{-1})

        Returns:
        --------
        array
            Time derivatives of perturbation state
        """
        k = args
        a  = jnp.exp(lna)
        aH = self.BG.aH(lna)
        metric_h   = y[0]
        metric_eta = y[1]

        # Metric perturbation derivatives
        sum_rho_delta = 0.
        sum_rho_plus_P_theta = 0.
        
        for i in range(len(self.perturbations_list)):
            species = self.perturbations_list[i]
            # If species has density perturbation, add to total.
            sum_rho_delta += species.rho_delta(lna, y, self.BG)
            # If species has velocity perturbation, add to total.
            sum_rho_plus_P_theta += species.rho_plus_P_theta(lna, y, self.BG)

        metric_h_prime   = 2./aH**2 * (k**2*metric_eta + 4.*jnp.pi*cnst.G*a**2/cnst.c_Mpc_over_s**2 * sum_rho_delta)
        metric_eta_prime = 4.*jnp.pi*cnst.G*a**2/aH/k**2 * sum_rho_plus_P_theta / cnst.c_Mpc_over_s**2

        # Now loop over all species and assemble their respective y_primes
        y_prime = jnp.array([metric_h_prime, metric_eta_prime])
        for i in range(len(self.perturbations_list)):
            species = self.perturbations_list[i]
            y_prime = jnp.concatenate((y_prime, species.y_prime(k, lna, metric_h_prime, metric_eta_prime, y, self.BG)))

        return y_prime

    @jax.named_scope("evolution_one_k_no_tca")
    def evolution_one_k(self, k):
        """
        Evolve perturbations for single wavenumber mode.

        Integrates Einstein-Boltzmann equations from early times through
        recombination to late times using adaptive time stepping.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})

        Returns:
        --------
        diffrax.Solution
            Dense solution object for interpolation

        """
        ### DIFFRAX INTEGRATION ###

        lna_start, lna_tca_off = self.get_tca_on_off(k) # Start and end times from tight coupling settings
        lna_end = 0.0

        # For small k's the superhorizon time can be set relatively late, but I impose a cutoff of z~20000 for all modes
        # at the very least.
        lna_start = jnp.minimum(lna_start,-10.0)
    
        # Initial conditions for tight coupling
        y_ini = self.initial_conditions_one_k(k, lna_start)

        # Settings for post-tight coupling
        term = diffrax.ODETerm(self.get_derivatives)
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(pcoeff=0.25, icoeff=0.80, dcoeff=0, rtol=1.e-3, atol=1.e-6) # DISCO-EB settings
        #stepsize_controller = diffrax.PIDController(rtol=1.e-5, atol=1.e-10)
        saveat = diffrax.SaveAt(dense=True)
        adjoint=diffrax.ForwardMode()

        sol = diffrax.diffeqsolve(
            term, solver,
            t0=lna_start, t1=lna_end, dt0=1.e-2, y0=y_ini,
            stepsize_controller=stepsize_controller,
            max_steps=2048,
            saveat=saveat,
            args=k,
            adjoint=adjoint
        )

        ### END OF DIFFRAX INTEGRATION ###

        return sol
        #return vmap(sol.evaluate)(lna)

    def evolution_one_k_scan(self, k, lna):
        """
        Evolve perturbations for single wavenumber mode.

        Integrates Einstein-Boltzmann equations from early times through
        recombination to late times using adaptive time stepping.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})

        Returns:
        --------
        diffrax.Solution
            Dense solution object for interpolation

        """
        ### DIFFRAX INTEGRATION ###

        lna_start, lna_tca_off = self.get_tca_on_off(k) # Start and end times from tight coupling settings
        lna_end = 0.0

        # For small k's the superhorizon time can be set relatively late, but I impose a cutoff of z~20000 for all modes
        # at the very least.
        #lna_start = jnp.minimum(lna_start, lna[0])
        lna_start = jnp.minimum(lna_start, -10.)
    
        # Initial conditions for tight coupling
        y_ini = self.initial_conditions_one_k(k, lna_start)

        # Settings for post-tight coupling
        term = diffrax.ODETerm(self.get_derivatives)
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(pcoeff=0.25, icoeff=0.80, dcoeff=0, rtol=1.e-3, atol=1.e-6) # DISCO-EB settings
        #stepsize_controller = diffrax.PIDController(rtol=1.e-3, atol=1.e-6)
        saveat = diffrax.SaveAt(dense=True)
        adjoint=diffrax.ForwardMode()

        sol = diffrax.diffeqsolve(
            term, solver,
            t0=lna_start, t1=lna_end, dt0=1.e-2, y0=y_ini,
            stepsize_controller=stepsize_controller,
            max_steps=2048,
            saveat=saveat,
            args=k,
            adjoint=adjoint
        )

        ### END OF DIFFRAX INTEGRATION ###

        return vmap(sol.evaluate)(lna)

    def make_output_table(self, k, lna, modes):
        """
        Create interpolatable perturbation table from evolution results.

        Extracts key perturbation modes and computes derived quantities.

        Parameters:
        -----------
        k : array
            Wavenumber grid (units: Mpc^{-1})
        lna : array
            Logarithm of scale factor grid
        modes : array
            Perturbation evolution results

        Returns:
        --------
        PerturbationTable
            Organized perturbation data for interpolation

        """
        CDM    = self.perturbations_list[-4]
        Baryon = self.perturbations_list[-3]
        Photon = self.perturbations_list[-2]

        # Shapes are (Nlna, Nk)
        metric_h   = modes[0]
        metric_eta = modes[1]
        delta_cdm  = modes[CDM.delta_idx]
        delta_b    = modes[Baryon.delta_idx]
        theta_b    = modes[Baryon.delta_idx+1]
        delta_g    = modes[Photon.delta_idx]
        theta_g    = modes[Photon.delta_idx+1]
        sigma_g    = modes[Photon.delta_idx+2]
        Gg0        = modes[Photon.delta_idx+Photon.num_F_ell_modes]
        Gg2        = modes[Photon.delta_idx+Photon.num_F_ell_modes+2]

        # Now the stuff that needs to be backwards calculated.
        karr = k[None, :]
        a  = jnp.exp(lna)[:, None]
        aH = self.BG.aH(lna)[:, None]
        cs2 = Baryon.cs2(lna, self.BG)[:, None]
        R = 4.*Photon.rho(lna, self.BG)[:, None]/3./Baryon.rho(lna, self.BG)[:, None]
        tau_c = self.BG.tau_c(lna)[:, None]

        # Baryon velocity derivative is needed for CMB
        theta_b_prime = -theta_b + cs2/aH*(karr**2*delta_b) + R/aH/tau_c*(theta_g-theta_b)

        # Sum of density and velocity perturbations over all species.
        # These are required again for the metric perturbation derivatives.
        sum_rho_delta = jnp.zeros_like(modes[0])
        sum_rho_plus_P_theta = jnp.zeros_like(modes[0])
        sum_rho_plus_P_sigma = jnp.zeros_like(modes[0])

        for s in self.perturbations_list:
            sum_rho_delta += vmap(s.rho_delta, in_axes=(0, 1, None))(lna, modes, self.BG)
            sum_rho_plus_P_theta += vmap(s.rho_plus_P_theta, in_axes=(0, 1, None))(lna, modes, self.BG)
            sum_rho_plus_P_sigma += vmap(s.rho_plus_P_sigma, in_axes=(0, 1, None))(lna, modes, self.BG)

        # Metric perturbation derivatives
        metric_h_prime = 2./aH**2 * (karr**2*metric_eta + 4.*jnp.pi*cnst.G*a**2/cnst.c_Mpc_over_s**2 * sum_rho_delta)
        metric_eta_prime = 4.*jnp.pi*cnst.G*a**2/aH * sum_rho_plus_P_theta / cnst.c_Mpc_over_s**2 / karr**2
        metric_alpha = aH*(metric_h_prime + 6.*metric_eta_prime)/2./ karr**2
        metric_alpha_prime = metric_eta/aH - 2.*metric_alpha \
                           - 12.*jnp.pi*cnst.G*a**2/aH * sum_rho_plus_P_sigma / cnst.c_Mpc_over_s**2 / karr**2


        return PerturbationTable(
            k,
            lna,
            delta_cdm,
            delta_b,
            theta_b,
            theta_b_prime,
            delta_g,
            theta_g,
            sigma_g,
            Gg0,
            Gg2,
            metric_h,
            metric_eta,
            metric_h_prime,
            metric_eta_prime,
            metric_alpha,
            metric_alpha_prime,
        )



class PerturbationTable(eqx.Module):
    """
    Interpolatable table of perturbation evolution.

    Stores perturbation modes as 2D arrays over wavenumber and time
    for efficient interpolation.

    Attributes:
    -----------
    k : array
        Wavenumber grid (units: Mpc^{-1})
    lna : array
        Logarithm of scale factor grid
    delta_cdm : array
        Cold dark matter density perturbations
    delta_b : array
        Baryon density perturbations
    theta_b : array
        Baryon velocity perturbations
    theta_b_prime : array
        Baryon velocity derivatives
    delta_g : array
        Photon density perturbations
    theta_g : array
        Photon velocity perturbations  
    sigma_g : array
        Photon quadrupole temperature moments
    Gg0 : array
        Photon monopole polarization moments
    Gg2 : array
        Photon quadrupole polarization moments
    metric_h : array
        Metric perturbation h
    metric_eta : array
        Metric perturbation η
    metric_h_prime : array
        Time derivative of metric h
    metric_eta_prime : array
        Time derivative of metric η
    metric_alpha : array
        Derived metric perturbation α
    metric_alpha_prime : array
        Time derivative of metric α
    """
    k         : jnp.array
    lna       : jnp.array
    delta_cdm : jnp.array
    delta_b   : jnp.array
    theta_b   : jnp.array
    theta_b_prime : jnp.array
    delta_g   : jnp.array
    theta_g   : jnp.array
    sigma_g   : jnp.array
    Gg0       : jnp.array
    Gg2       : jnp.array

    metric_h       : jnp.array
    metric_eta     : jnp.array
    metric_h_prime : jnp.array
    metric_eta_prime : jnp.array
    metric_alpha   : jnp.array
    metric_alpha_prime : jnp.array

class MockPerturbationTable(PerturbationTable):

    def __init__(self):
        data = np.load(file_dir+"/../Module_Tests/perturbations.npz")

        super().__init__(
            k=jnp.array(data["k"]),
            lna=jnp.array(data["lna"]),
            delta_cdm=jnp.array(data["delta_cdm"]),
            delta_b=jnp.array(data["delta_b"]),
            theta_b=jnp.array(data["theta_b"]),
            theta_b_prime=jnp.array(data["theta_b_prime"]),
            delta_g=jnp.array(data["delta_g"]),
            theta_g=jnp.array(data["theta_g"]),
            sigma_g=jnp.array(data["sigma_g"]),
            Gg0=jnp.array(data["Gg0"]),
            Gg2=jnp.array(data["Gg2"]),
            metric_h=jnp.array(data["metric_h"]),
            metric_eta=jnp.array(data["metric_eta"]),
            metric_h_prime=jnp.array(data["metric_h_prime"]),
            metric_eta_prime=jnp.array(data["metric_eta_prime"]),
            metric_alpha=jnp.array(data["metric_alpha"]),
            metric_alpha_prime=jnp.array(data["metric_alpha_prime"]),
        )