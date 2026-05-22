import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, lax
import diffrax
from diffrax import with_stepsize_controller_tols
from diffrax._root_finder import VeryChord
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

    Attributes:
    -----------
    species_list : tuple  
        A list of all fluids in the cosmology
    species_dict : dict 
        A dictionary containing the names of all fluids, in the same order as 
        they appear in species_list.
    k_axis_perturbations : jnp.array
        A list of wavenumbers k at which to compute perturbations
    specs : dict 
        A dictionary containing run options
    adjoint : diffrax.adjoint
        Adjoint mode for diffrax solves.  Default is ForwardMode.

    Methods:
    --------
    full_evolution : Evolve perturbations for multiple k modes
    evolution_one_k : Evolve perturbations for single k mode
    get_tca_on_off : Determine tight coupling approximation times
    initial_conditions_one_k : Compute initial perturbation conditions
    get_derivatives : Compute perturbation time derivatives
    make_output_table : Create interpolatable perturbation table
    """

    species_list : tuple
    species_dict : dict
    k_axis_perturbations : jnp.array
    specs : dict

    adjoint : "diffrax.adjoint" = eqx.field(static=True)

    ### RSA ###
    k_RSA_off : jnp.array
    k_RSA_on  : jnp.array
    k_RSA_break = 0.09 # Hard coded, all higher k modes get RSA treatment.

    lna_RSA_break = -4.0 # Hard coded, evolution for k>k_RSA_break, lna > lna_RSA_break uses RSA.
    lna_axis_perturbations = jnp.linspace(-8.0, 0.0, 550) # Hard coded.
    lna_RSA_off = lna_axis_perturbations[lna_axis_perturbations < lna_RSA_break] 
    lna_RSA_on  = lna_axis_perturbations[lna_axis_perturbations >= lna_RSA_break] 
    
    def __init__(
        self,
        species_list,
        species_dict,
        k_axis_perturbations=jnp.geomspace(1.e-4, 0.4, 600),
        specs = {},
        adjoint = diffrax.ForwardMode,
    ):
        self.species_list = species_list
        self.species_dict = species_dict
        self.k_axis_perturbations = k_axis_perturbations
        self.specs = specs
        self.adjoint = adjoint

        self.k_RSA_off = self.k_axis_perturbations[self.k_axis_perturbations < self.k_RSA_break]
        self.k_RSA_on  = self.k_axis_perturbations[self.k_axis_perturbations >= self.k_RSA_break]

    def full_evolution(self, args):
        """
        Evolve perturbations for multiple wavenumber modes.

        Integrates perturbation equations for a range of k modes,
        then interpolates results onto common time grid.

        Parameters:
        -----------
        k    : jnp.array
            1D axis of wavenumbers k. Perturbations are computed and stored at these values.
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        PerturbationTable
            Interpolatable table of perturbation evolution

        Notes:
        ------
        Uses logarithmic k spacing from 10^-4 to ~0.5 Mpc^-1 with 100 points.
        Time integration runs from early times to z=1 (lna=-ln(2)).
        """
        BG, params = args
        lna = jnp.linspace(BG.lna_transfer_start,  0., 500)

        # This scan function is only used if on CPU.
        # For GPUs we vmap over the wavenumbers instead
        def scan_fun(_, ki):
            # evolution_one_k returns shape (Nlna, Ny)
            y = self.evolution_one_k(ki, lna, args) 
            return None, y

        if jax.default_backend() =='gpu':
            res, _ = vmap(self.evolution_one_k,in_axes=[0,None,None])(self.k_axis_perturbations, self.lna_axis_perturbations, args)
        else: 
            _, res = lax.scan(scan_fun, None, self.k_axis_perturbations)      # res has shape (Nk, Nlna, Ny)

        res = res.transpose(2, 1, 0) # Transpose so the shape is (Ny, Nlna, Nk), easier for vmapping over in PT

        PT = self.make_output_table(self.lna_axis_perturbations, res, args)
        return PT

    def full_evolution_RSA(self, args):
        """
        Evolve perturbations for multiple wavenumber modes.

        Integrates perturbation equations for a range of k modes,
        then interpolates results onto common time grid.

        Parameters:
        -----------
        k    : jnp.array
            1D axis of wavenumbers k. Perturbations are computed and stored at these values.
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        PerturbationTable
            Interpolatable table of perturbation evolution

        Notes:
        ------
        Uses logarithmic k spacing from 10^-4 to ~0.5 Mpc^-1 with 100 points.
        Time integration runs from early times to z=1 (lna=-ln(2)).
        """
        BG, params = args
        lna = jnp.linspace(BG.lna_transfer_start,  0., 500)

        # This scan function is only used if on CPU.
        # For GPUs we vmap over the wavenumbers instead
        def scan_fun(_, ki):
            # evolution_one_k returns shape (Nlna, Ny)
            y = self.evolution_one_k(ki, lna, args) 
            return None, y

        if jax.default_backend() =='gpu':
            res_no_RSA, _ = vmap(self.evolution_one_k,in_axes=[0,None,None])(self.k_RSA_off, self.lna_axis_perturbations, args)
            res_RSA, _ , _  = vmap(self.evolution_one_k_RSA,in_axes=[0, None])(self.k_RSA_on, args)
            res = jnp.concatenate((res_no_RSA, res_RSA))
        else: 
            _, res = lax.scan(scan_fun, None, self.k_axis_perturbations)      # res has shape (Nk, Nlna, Ny)

        res = res.transpose(2, 1, 0) # Transpose so the shape is (Ny, Nlna, Nk), easier for vmapping over in PT

        PT = self.make_output_table(self.lna_axis_perturbations, res, args)
        return PT

    def get_starting_time(self, k, args):
        """
        Determine tight coupling approximation time range.

        Finds start and end times for tight coupling between photons and baryons
        by computing when Thomson scattering becomes ineffective relative to
        Hubble and horizon crossing time scales.

        Parameters:
        -----------
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        tuple
            (lna_start, lna_end) for tight coupling period

        Notes:
        ------
        Uses thresholds: τc/τh < 0.0015 (start), τh/τk < 0.07 (start),
        τc/τh > 0.015 (end), τc/τk > 0.01 (end).
        """
        BG, params = args

        # 1) Starting lna
        lna_start_range = jnp.linspace(-20.0, -10.0, 10000)

        # a) τc/τh  →  f1(lna) = BG.tau_c * BG.aH
        f1 = BG.tau_c(lna_start_range, params) * BG.aH(lna_start_range, params)
        # invert f1(lna) = thr1  →  lna = interp(thr1, f1, lna_range)
        lna1 = jnp.interp(self.specs["R_tc"], f1, lna_start_range)    # jnp.interp ends up being 
                                                        # faster than fast_interp through here
        # b) τh/τk  →  f2(lna) = k / BG.aH
        f2 = k / BG.aH(lna_start_range, params)
        # invert f2(lna) = thr2
        lna2 = jnp.interp(self.specs["R_large"], f2, lna_start_range)

        lna_ini = jnp.minimum(lna1, lna2)

        return lna_ini

    def initial_conditions_one_k(self, k, lna_ini, args):
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
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        array
            Initial perturbation state vector

        Notes:
        ------
        Uses CLASS-style initial conditions with metric perturbations h and η.
        Assumes adiabatic initial conditions with vanishing isocurvature modes.
        """
        BG, params = args
        ### CLASS Initial Conditions ###
        a = jnp.exp(lna_ini)
        tau_ini = BG.tau(lna_ini)
        
        om = params["om"]

        metric_eta_ini = (1.-k**2*tau_ini**2/12./(15.+4.*params['R_nu'])*(5.+4.*params['R_nu'] - (16.*params['R_nu']*params['R_nu']+280.*params['R_nu']+325)/10./(2.*params['R_nu']+15.)*tau_ini*om))

        all_fluid_ini = jnp.concatenate([p.y_ini(k, tau_ini, params) for p in self.species_list])
        y_ini = jnp.concatenate((jnp.array([metric_eta_ini]), all_fluid_ini))
        
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
        args : tuple
            Wavenumber k and background cosmology (k, BG, params)

        Returns:
        --------
        array
            Time derivatives of perturbation state
        """
        k, BG, params = args
        a  = jnp.exp(lna)
        aH = BG.aH(lna, params)
        metric_eta = y[0]

        # Metric perturbation derivatives
        sum_rho_delta = 0.
        sum_rho_plus_P_theta = 0.
        
        for i in range(len(self.species_list)):
            species = self.species_list[i]
            # If species has density perturbation, add to total.
            sum_rho_delta += species.rho_delta(lna, y, params)
            # If species has velocity perturbation, add to total.
            sum_rho_plus_P_theta += species.rho_plus_P_theta(lna, y, params)

        metric_h_prime   = 2./aH**2 * (k**2*metric_eta + 4.*jnp.pi*cnst.G*a**2/cnst.c_Mpc_over_s**2 * sum_rho_delta)
        metric_eta_prime = 4.*jnp.pi*cnst.G*a**2/aH/k**2 * sum_rho_plus_P_theta / cnst.c_Mpc_over_s**2

        # Now loop over all species and assemble their respective y_primes
        args = (BG, params, self.species_list, self.species_dict)
        y_prime = jnp.array([metric_eta_prime])
        for i in range(len(self.species_list)):
            species = self.species_list[i]
            y_prime = jnp.concatenate((y_prime, species.y_prime(k, lna, metric_h_prime, metric_eta_prime, y, args)))

        return y_prime

    def get_derivatives_RSA(self, lna, y, args):
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
        args : tuple
            Wavenumber k and background cosmology (k, BG, params)

        Returns:
        --------
        array
            Time derivatives of perturbation state
        """
        k, BG, params = args
        a  = jnp.exp(lna)
        aH = BG.aH(lna, params)
        metric_eta = y[0]

        # Metric perturbation derivatives
        sum_rho_delta = 0.
        sum_rho_plus_P_theta = 0.
        
        for s in self.species_list:
            # Simply skip radiation
            if s.name != "Photon" and s.name != "MasslessNeutrino":
                # If species has density perturbation, add to total.
                sum_rho_delta += s.rho_delta(lna, y, params)
                # If species has velocity perturbation, add to total.
                sum_rho_plus_P_theta += s.rho_plus_P_theta(lna, y, params)

        metric_h_prime   = 2./aH**2 * (k**2*metric_eta + 4.*jnp.pi*cnst.G*a**2/cnst.c_Mpc_over_s**2 * sum_rho_delta)
        metric_eta_prime = 4.*jnp.pi*cnst.G*a**2/aH/k**2 * sum_rho_plus_P_theta / cnst.c_Mpc_over_s**2

        # Now loop over all species and assemble their respective y_primes
        args = (BG, params, self.species_list, self.species_dict)
        y_prime = jnp.array([metric_eta_prime])
        for s in self.species_list:
            # Radiation now gets trivial zero derivative.
            if s.name == "Photon" or s.name == "MasslessNeutrino":
                y_prime = jnp.concatenate((y_prime, jnp.zeros(s.num_equations)))
            else:
                y_prime = jnp.concatenate((y_prime, s.y_prime(k, lna, metric_h_prime, metric_eta_prime, y, args)))

        return y_prime

    def evolution_one_k(self, k, lna, args):
        """
        Evolve perturbations for single wavenumber mode.

        Integrates Einstein-Boltzmann equations from early times through
        recombination to late times using adaptive time stepping.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        lna : array
            Logarithm of scale factor grid for output
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        diffrax.Solution
            Dense solution object for interpolation

        """
        ### DIFFRAX INTEGRATION ###

        lna_start = self.get_starting_time(k, args) # Start and end times from tight coupling settings
        lna_end = lna[-1] # End at the last lna requested. 

        # For small k's the superhorizon time can be set relatively late, but we impose a of lna_min = -8.0, 
        # so that initial condition don't start later than the first number of saveat.
        lna_start = jnp.minimum(lna_start, self.lna_axis_perturbations[0])
    
        # Initial conditions for tight coupling
        y_ini = self.initial_conditions_one_k(k, lna_start, args)

        # Settings for post-tight coupling
        term = diffrax.ODETerm(self.get_derivatives)
        # LCDM defaults for very high k sometimes failed with reverseAD.
        # The reason for that was the default precision parameter in
        # the Kvaerno5 rootfinder (via VeryChord).  Parameter now read
        # in from specs; if reverse-AD produces NaNs (especially on omega_b/
        # omega_cdm), decrease kappa_PE in specs (e.g. to 1e-3) to tighten 
        # convergence.
        _rf = eqx.tree_at(
            lambda s: s.kappa,
            with_stepsize_controller_tols(VeryChord)(),
            replace=self.specs["kappa_PE"],
        )
        solver = diffrax.Kvaerno5(root_finder=_rf)

        rtol=jnp.where(
            k > self.specs["k_split_PE"],
            self.specs["rtol_large_k_PE"],
            self.specs["rtol_small_k_PE"]
        )

        atol=jnp.where(
            k > self.specs["k_split_PE"],
            self.specs["atol_large_k_PE"],
            self.specs["atol_small_k_PE"]
        )

        stepsize_controller = diffrax.PIDController(pcoeff=self.specs["pcoeff_PE"], icoeff=self.specs["icoeff_PE"], dcoeff=self.specs["dcoeff_PE"], rtol=rtol, atol=atol)
        saveat = diffrax.SaveAt(ts=lna)
        adjoint=self.adjoint()

        sol = diffrax.diffeqsolve(
            term, solver,
            t0=lna_start, t1=lna_end, dt0=1.e-2, y0=y_ini,
            stepsize_controller=stepsize_controller,
            max_steps=self.specs["max_steps_PE"],
            saveat=saveat,
            args=(k,*args),
            adjoint=adjoint
        )

        ### END OF DIFFRAX INTEGRATION ###

        return sol.ys, sol.stats

    def evolution_one_k_RSA(self, k, args):
        """
        Perturbation evolution for large k values that use RSA at late times.
        First get the early time solution using evolution_one_k with the full equations.
        Then at lna_RSA_break, switch to another solve using the abbreviated RSA equations.
        """

        # Get pre RSA solution
        res_RSA_off, sol_RSA_off_stats = self.evolution_one_k(k, self.lna_RSA_off, args)

        # Set initial conditions
        y_ini = res_RSA_off[-1]
        photon = self.species_list[self.species_dict["Photon"]]
        nu = self.species_list[self.species_dict["MasslessNeutrino"]]

        # Zeroth order, set all radiation perturbations to zero.
        y_ini = y_ini.at[photon.first_idx:photon.first_idx+photon.num_equations].set(jnp.zeros(photon.num_equations))
        y_ini = y_ini.at[nu.first_idx:nu.first_idx+nu.num_equations].set(jnp.zeros(nu.num_equations))

        # Diffrax settings
        _rf = eqx.tree_at(
            lambda s: s.kappa,
            with_stepsize_controller_tols(VeryChord)(),
            replace=self.specs["kappa_PE"],
        )
        solver = diffrax.Kvaerno5(root_finder=_rf)

        rtol=jnp.where(
            k > self.specs["k_split_PE"],
            self.specs["rtol_large_k_PE"],
            self.specs["rtol_small_k_PE"]
        )

        atol=jnp.where(
            k > self.specs["k_split_PE"],
            self.specs["atol_large_k_PE"],
            self.specs["atol_small_k_PE"]
        )
        stepsize_controller = diffrax.PIDController(pcoeff=self.specs["pcoeff_PE"], icoeff=self.specs["icoeff_PE"], dcoeff=self.specs["dcoeff_PE"], rtol=rtol, atol=atol)
        saveat = diffrax.SaveAt(ts=self.lna_RSA_on)
        adjoint=self.adjoint()

        # Differential equations with RSA
        term = diffrax.ODETerm(self.get_derivatives_RSA)
        sol_RSA_on = diffrax.diffeqsolve(
            term, solver,
            t0=self.lna_RSA_off[-1], t1=self.lna_RSA_on[-1], dt0=1.e-2, y0=y_ini,
            stepsize_controller=stepsize_controller,
            max_steps=self.specs["max_steps_PE"],
            saveat=saveat,
            args=(k,*args),
            adjoint=adjoint
        )

        return jnp.concatenate((res_RSA_off, sol_RSA_on.ys)), sol_RSA_on.stats, sol_RSA_off_stats


    def make_output_table(self, lna, modes, args):
        """
        Create interpolatable perturbation table from evolution results.

        Extracts key perturbation modes and computes derived quantities.

        Parameters:
        -----------
        lna : array
            Logarithm of scale factor grid
        modes : array
            Perturbation evolution results
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        PerturbationTable
            Organized perturbation data for interpolation

        """
        k = self.k_axis_perturbations
        BG, params = args

        metric_eta = modes[0]

        # Build per-species perturbation dicts first; theta_b_prime draws from them.
        species_perturbations = {
            s.name: s.output_perturbations(lna, modes, (BG, params))
            for s in self.species_list
        }

        # Baryon velocity derivative — backward-calculated from the Boltzmann equations.
        # Requires Baryon and Photon objects for cs2 and the photon-baryon coupling R.
        Baryon = self.species_list[self.species_dict["Baryon"]]
        Photon = self.species_list[self.species_dict["Photon"]]
        delta_b = species_perturbations["Baryon"]["delta"]
        theta_b = species_perturbations["Baryon"]["theta"]
        theta_g = species_perturbations["Photon"]["theta"]

        karr  = k[None, :]
        a     = jnp.exp(lna)[:, None]
        aH    = BG.aH(lna, params)[:, None]
        cs2   = Baryon.cs2(lna, (BG, params, self.species_list, self.species_dict))[:, None]
        R     = 4.*Photon.rho(lna, params)[:, None]/3./Baryon.rho(lna, params)[:, None]
        tau_c = BG.tau_c(lna, params)[:, None]

        theta_b_prime = -theta_b + cs2/aH*(karr**2*delta_b) + R/aH/tau_c*(theta_g-theta_b)

        # Sum density/velocity/shear over all species for metric derivatives and delta_m.
        sum_rho_delta        = jnp.zeros_like(modes[0])
        sum_rho_plus_P_theta = jnp.zeros_like(modes[0])
        sum_rho_plus_P_sigma = jnp.zeros_like(modes[0])
        sum_rho_delta_m      = jnp.zeros_like(modes[0])
        sum_rho_m            = 0.

        for s in self.species_list:
            if s.num_equations > 0:
                rho_delta = vmap(s.rho_delta, in_axes=(0, 1, None))(lna, modes, params)
                sum_rho_delta        += rho_delta
                sum_rho_plus_P_theta += vmap(s.rho_plus_P_theta, in_axes=(0, 1, None))(lna, modes, params)
                sum_rho_plus_P_sigma += vmap(s.rho_plus_P_sigma, in_axes=(0, 1, None))(lna, modes, params)

                if s.is_matter:
                    sum_rho_delta_m += rho_delta
                    sum_rho_m       += s.rho(lna, params)

        delta_m = sum_rho_delta_m / sum_rho_m[:, None]

        metric_h_prime     = 2./aH**2 * (karr**2*metric_eta + 4.*jnp.pi*cnst.G*a**2/cnst.c_Mpc_over_s**2 * sum_rho_delta)
        metric_eta_prime   = 4.*jnp.pi*cnst.G*a**2/aH * sum_rho_plus_P_theta / cnst.c_Mpc_over_s**2 / karr**2
        metric_alpha       = aH*(metric_h_prime + 6.*metric_eta_prime)/2./karr**2
        metric_alpha_prime = metric_eta/aH - 2.*metric_alpha \
                           - 12.*jnp.pi*cnst.G*a**2/aH * sum_rho_plus_P_sigma / cnst.c_Mpc_over_s**2 / karr**2

        return PerturbationTable(
            k,
            lna,
            delta_m,
            theta_b_prime,
            metric_eta,
            metric_h_prime,
            metric_eta_prime,
            metric_alpha,
            metric_alpha_prime,
            species_perturbations,
        )

class PerturbationTable(eqx.Module):
    """
    Interpolatable table of perturbation evolution.

    Stores perturbation modes as 2D arrays over wavenumber and time
    for efficient interpolation. Per-species perturbations in physically
    meaningful form are accessible via species_perturbations.

    Attributes:
    -----------
    k : array
        Wavenumber grid (units: Mpc^{-1})
    lna : array
        Logarithm of scale factor grid
    delta_m : array
        Total matter density perturbation, weighted sum over all matter species
    theta_b_prime : array
        Baryon velocity derivative (backward-calculated from Boltzmann equations)
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
    species_perturbations : dict
        Named perturbation arrays for each species, keyed by species name.
        Each value is a dict {quantity: array(Nlna, Nk)}.
        Species with no perturbations (e.g. dark energy) map to {}.
    """
    k             : jnp.array
    lna           : jnp.array
    delta_m       : jnp.array
    theta_b_prime : jnp.array

    metric_eta         : jnp.array
    metric_h_prime     : jnp.array
    metric_eta_prime   : jnp.array
    metric_alpha       : jnp.array
    metric_alpha_prime : jnp.array

    species_perturbations : dict