import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, lax
import diffrax
import equinox as eqx
from interpax import CubicSpline

from . import constants as cnst
from . import ABCMBTools as tools
from .spectrum import bessel_l_tab, xphi0_tab, phi0_tab, xphi1_tab, phi1_tab, \
    xphi2_tab, phi2_tab, j

jax.config.update("jax_enable_x64", True)

"""
Tensor (primordial gravitational wave) perturbation module.

Evolves the tensor metric perturbation h and the photon / massless-neutrino
tensor Boltzmann hierarchies, and computes the tensor contributions to the
CMB angular power spectra (TT, TE, EE, BB). All equations, conventions and
default settings are transcribed from CLASS (synchronous gauge, flat space,
gw_ini = 1 normalization with P_h(k) = r A_s (k/k_pivot)^{n_t}).
"""

SQRT6 = jnp.sqrt(6.)


def get_tensor_k_axes(specs, k_axis_perturbations, k_axis_transfer):
    """
    Truncate the scalar k grids at the tensor k_max.

    CLASS builds the tensor k grid with the same stepping formula as the
    scalar one but stops at k_max = k_max_tau0_over_l_max * l_tensor_max /
    tau0, so truncating the scalar grids reproduces it exactly (one extra
    point is kept past k_max, matching CLASS's final loop iteration).

    Parameters:
    -----------
    specs : dict
        Run options (uses l_tensor_max, k_max_tau0_over_l_max, tau0_fid)
    k_axis_perturbations : array
        Scalar perturbation k grid (units: Mpc^{-1})
    k_axis_transfer : array
        Scalar transfer-integration k grid (units: Mpc^{-1})

    Returns:
    --------
    tuple
        (k_axis_perturbations_tensor, k_axis_transfer_tensor)
    """
    k_max_tensor = specs["k_max_tau0_over_l_max"] * specs["l_tensor_max"] \
        / specs["tau0_fid"]

    kp = np.asarray(k_axis_perturbations)
    kt = np.asarray(k_axis_transfer)
    ip = np.searchsorted(kp, k_max_tensor)
    it = np.searchsorted(kt, k_max_tensor)

    return (
        jnp.array(kp[:min(ip + 1, len(kp))]),
        jnp.array(kt[:min(it + 1, len(kt))]),
    )


class TensorSourceTable(eqx.Module):
    """
    Interpolatable table of tensor source functions.

    Attributes:
    -----------
    k : array
        Wavenumber grid (units: Mpc^{-1})
    lna : array
        Logarithm of scale factor grid
    source_T2 : array, shape (Nlna, Nk)
        Tensor temperature source -h' exp(-kappa) + g Pi (conformal-time h')
    source_E : array, shape (Nlna, Nk)
        Tensor polarization source sqrt(6) g Pi (CLASS/CAMB sign convention)
    """
    k         : jnp.array
    lna       : jnp.array
    source_T2 : jnp.array
    source_E  : jnp.array


class TensorPerturbationEvolver(eqx.Module):
    """
    Linear tensor perturbation evolution solver.

    Evolves the gravitational wave amplitude h together with the photon
    tensor temperature/polarization hierarchies and a massless-neutrino
    tensor hierarchy, in synchronous gauge. Mirrors CLASS's tensor sector
    under its default tensor_method = massless_approximation, where the
    neutrino hierarchy density is rho(massless nu) + 3 P(massive nu).

    State vector (CLASS variable conventions):
        [delta_g, theta_g, shear_g, F3..F_{l_max_g_ten},
         G0..G_{l_max_pol_g_ten},
         delta_ur, theta_ur, shear_ur, F3ur..F_{l_max_ur_ten},
         h, hdot]
    with delta_g = F0, theta_g = (3k/4) F1, shear_g = F2/2, and hdot the
    conformal-time derivative of h.

    Attributes:
    -----------
    species_list : tuple
        A list of all fluids in the cosmology
    species_dict : dict
        A dictionary containing the names of all fluids, in the same order
        as they appear in species_list.
    k_axis_tensor : jnp.array
        Wavenumbers k at which tensor perturbations are computed
    specs : dict
        A dictionary containing run options
    adjoint : diffrax.adjoint
        Adjoint mode for diffrax solves. Default is ForwardMode.

    Methods:
    --------
    full_evolution : Evolve tensor perturbations for multiple k modes
    evolution_one_k : Evolve tensor perturbations for single k mode
    get_starting_time : Determine integration start time
    initial_conditions_one_k : Compute initial tensor perturbation conditions
    get_derivatives : Compute tensor perturbation time derivatives
    make_output_table : Create tensor source function table
    """

    species_list  : tuple
    species_dict  : dict
    k_axis_tensor : jnp.array
    specs         : dict

    num_F : int = eqx.field(static=True)
    num_G : int = eqx.field(static=True)
    num_U : int = eqx.field(static=True)

    adjoint : "diffrax.adjoint" = eqx.field(static=True)

    def __init__(
        self,
        species_list,
        species_dict,
        k_axis_tensor,
        specs={},
        adjoint=diffrax.ForwardMode,
    ):
        self.species_list = species_list
        self.species_dict = species_dict
        self.k_axis_tensor = k_axis_tensor
        self.specs = specs
        self.num_F = specs["l_max_g_ten"] + 1
        self.num_G = specs["l_max_pol_g_ten"] + 1
        self.num_U = specs["l_max_ur_ten"] + 1
        self.adjoint = adjoint

    def rho_relativistic(self, lna, params):
        """
        Relativistic density driving the neutrino tensor hierarchy.

        Matches CLASS tensor_method = massless_approximation: massless
        neutrinos contribute rho, massive neutrinos contribute 3 P (their
        relativistic part). Custom species may opt in by defining a
        ``tensor_rho_rel(lna, params)`` method; species without one (and
        without "neutrino" in their name) are not included, as in CLASS.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        params : dict
            Cosmological parameters

        Returns:
        --------
        float
            Relativistic energy density (units: eV cm^{-3})
        """
        rho = 0.
        for s in self.species_list:
            if hasattr(s, "tensor_rho_rel"):
                rho += s.tensor_rho_rel(lna, params)
            elif s.name == "MasslessNeutrino":
                rho += s.rho(lna, params)
            elif s.name == "MassiveNeutrino":
                rho += 3. * s.P(lna, params)
        return rho

    def get_starting_time(self, k, args):
        """
        Determine integration start time for one tensor mode.

        Same criteria as the scalar evolver (PerturbationEvolver
        .get_starting_time): start when Thomson scattering is efficient
        relative to the Hubble time AND the mode is super-horizon.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        float
            Starting log scale factor
        """
        BG, params = args

        lna_start_range = jnp.linspace(-20.0, -10.0, 10000)

        f1 = BG.tau_c(lna_start_range, params) * BG.aH(lna_start_range, params)
        lna1 = jnp.interp(self.specs["R_tc"], f1, lna_start_range)

        f2 = k / BG.aH(lna_start_range, params)
        lna2 = jnp.interp(self.specs["R_large"], f2, lna_start_range)

        return jnp.minimum(lna1, lna2)

    def initial_conditions_one_k(self, k, lna_ini, args):
        """
        Compute initial conditions for tensor perturbation evolution.

        The GW amplitude is frozen super-horizon: h = 1/sqrt(6) (CLASS
        gw_ini = 1 normalization, with the primordial spectrum carrying
        r A_s), hdot = 0, and all photon/neutrino tensor moments zero.

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
            Initial tensor perturbation state vector
        """
        Ny = self.num_F + self.num_G + self.num_U + 2
        y = jnp.zeros(Ny)
        y = y.at[-2].set(1. / SQRT6)
        return y

    def get_derivatives(self, lna, y, args):
        """
        Compute time derivatives for tensor perturbation evolution.

        CLASS tensor equations (perturbations.c, flat space) divided by aH
        to integrate in lna. kappa' = 1/tau_c is the Thomson scattering
        rate; the GW equation is h'' = -2 aH h' - k^2 h + S with S the
        tensor anisotropic stress of photons and relativistic neutrinos.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Current tensor perturbation state vector
        args : tuple
            Wavenumber k, background cosmology and parameters (k, BG, params)

        Returns:
        --------
        array
            Time derivatives of tensor perturbation state (d/dlna)
        """
        k, BG, params = args
        a = jnp.exp(lna)
        aH = BG.aH(lna, params)
        tau = BG.tau(lna)
        tau_c = BG.tau_c(lna, params)

        NF, NG, NU = self.num_F, self.num_G, self.num_U
        F = y[0:NF]
        G = y[NF:NF + NG]
        U = y[NF + NG:NF + NG + NU]
        h = y[-2]
        hdot = y[-1]

        delta_g, theta_g, shear_g = F[0], F[1], F[2]

        # Pi^(2), the polarization+temperature quadrupole combination
        P2 = -1. / SQRT6 * (
            1. / 10. * delta_g
            + 2. / 7. * shear_g
            + 3. / 70. * F[4]
            - 3. / 5. * G[0]
            + 6. / 7. * G[2]
            - 3. / 70. * G[4]
        )

        # Photon tensor temperature hierarchy
        delta_g_prime = (-4. / 3. * theta_g - (delta_g + SQRT6 * P2) / tau_c
                         + SQRT6 * hdot) / aH
        theta_g_prime = (k**2 * (delta_g / 4. - shear_g) - theta_g / tau_c) / aH
        shear_g_prime = (4. / 15. * theta_g - 3. / 10. * k * F[3]
                         - shear_g / tau_c) / aH
        F3_prime = (k / 7. * (6. * shear_g - 4. * F[4]) - F[3] / tau_c) / aH

        Flmax = NF - 1
        L = jnp.arange(4, Flmax)
        Fl_prime = (k / (2. * L + 1.) * (L * F[L - 1] - (L + 1.) * F[L + 1])
                    - F[L] / tau_c) / aH
        Flmax_prime = (k * F[Flmax - 1] - (Flmax + 1.) / tau * F[Flmax]
                       - F[Flmax] / tau_c) / aH

        # Photon tensor polarization hierarchy
        G0_prime = (-k * G[1] - (G[0] - SQRT6 * P2) / tau_c) / aH
        Glmax = NG - 1
        L = jnp.arange(1, Glmax)
        Gl_prime = (k / (2. * L + 1.) * (L * G[L - 1] - (L + 1.) * G[L + 1])
                    - G[L] / tau_c) / aH
        Glmax_prime = (k * G[Glmax - 1] - (Glmax + 1.) / tau * G[Glmax]
                       - G[Glmax] / tau_c) / aH

        # Massless-neutrino tensor hierarchy (no scattering)
        delta_u, theta_u, shear_u = U[0], U[1], U[2]
        delta_u_prime = (-4. / 3. * theta_u) / aH + SQRT6 * hdot / aH
        theta_u_prime = k**2 * (delta_u / 4. - shear_u) / aH
        shear_u_prime = (4. / 15. * theta_u - 3. / 10. * k * U[3]) / aH
        U3_prime = k / 7. * (6. * shear_u - 4. * U[4]) / aH

        Ulmax = NU - 1
        L = jnp.arange(4, Ulmax)
        Ul_prime = k / (2. * L + 1.) * (L * U[L - 1] - (L + 1.) * U[L + 1]) / aH
        Ulmax_prime = (k * U[Ulmax - 1] - (Ulmax + 1.) / tau * U[Ulmax]) / aH

        # GW equation. CLASS units: rho_class = (8 pi G / 3 c^2) rho_phys,
        # gw_source = -sqrt(6) * 4 a^2 * sum_i rho_class,i *
        #             (delta_i/15 + 4 shear_i/21 + F4_i/35)
        i = self.species_dict["Photon"]
        rho_g = self.species_list[i].rho(lna, params)
        rho_u = self.rho_relativistic(lna, params)
        rho_unit = 8. * jnp.pi * cnst.G / 3. / cnst.c_Mpc_over_s**2

        gw_source = -SQRT6 * 4. * a**2 * rho_unit * (
            rho_g * (delta_g / 15. + 4. / 21. * shear_g + F[4] / 35.)
            + rho_u * (delta_u / 15. + 4. / 21. * shear_u + U[4] / 35.)
        )

        h_prime = hdot / aH
        hdot_prime = (-2. * aH * hdot - k**2 * h + gw_source) / aH

        return jnp.concatenate((
            jnp.array([delta_g_prime, theta_g_prime, shear_g_prime, F3_prime]),
            Fl_prime, jnp.array([Flmax_prime]),
            jnp.array([G0_prime]), Gl_prime, jnp.array([Glmax_prime]),
            jnp.array([delta_u_prime, theta_u_prime, shear_u_prime, U3_prime]),
            Ul_prime, jnp.array([Ulmax_prime]),
            jnp.array([h_prime, hdot_prime]),
        ))

    def evolution_one_k(self, k, lna, args):
        """
        Evolve tensor perturbations for single wavenumber mode.

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
        array
            Tensor perturbation state at the requested lna values
        """
        lna_start = self.get_starting_time(k, args)
        lna_start = jnp.minimum(lna_start, -10.)

        y_ini = self.initial_conditions_one_k(k, lna_start, args)

        term = diffrax.ODETerm(self.get_derivatives)
        solver = diffrax.Kvaerno5()

        # Uniform tolerances, tighter than the scalar PE defaults: the
        # scalar large-k rtol (1e-4) biases tensor BB low by ~0.7% at
        # l~450 (accumulated solver amplitude error grows with k). The
        # defaults reproduce the fully converged answer to ~1e-4.
        # See specs rtol_ten / atol_ten.
        stepsize_controller = diffrax.PIDController(
            pcoeff=self.specs["pcoeff_PE"],
            icoeff=self.specs["icoeff_PE"],
            dcoeff=self.specs["dcoeff_PE"],
            rtol=self.specs.get("rtol_ten", 1.e-5),
            atol=self.specs.get("atol_ten", 1.e-9)
        )
        saveat = diffrax.SaveAt(ts=lna)

        sol = diffrax.diffeqsolve(
            term, solver,
            t0=lna_start, t1=0.0, dt0=1.e-2, y0=y_ini,
            stepsize_controller=stepsize_controller,
            max_steps=self.specs.get("max_steps_ten", 4096),
            saveat=saveat,
            args=(k, *args),
            adjoint=self.adjoint()
        )

        return sol.ys

    def full_evolution(self, args):
        """
        Evolve tensor perturbations for all wavenumber modes.

        Parameters:
        -----------
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        TensorSourceTable
            Table of tensor source functions on the (lna, k) grid
        """
        BG, params = args
        lna = jnp.linspace(BG.lna_transfer_start, 0.,
                           self.specs.get("Nlna_ten", 500))

        def scan_fun(_, ki):
            y = self.evolution_one_k(ki, lna, args)
            return None, y

        if jax.default_backend() == 'gpu':
            res = vmap(self.evolution_one_k, in_axes=[0, None, None])(
                self.k_axis_tensor, lna, args)
        else:
            _, res = lax.scan(scan_fun, None, self.k_axis_tensor)

        res = res.transpose(2, 1, 0)  # (Ny, Nlna, Nk)

        return self.make_output_table(lna, res, args)

    def make_output_table(self, lna, modes, args):
        """
        Create tensor source function table from evolution results.

        Computes the CLASS tensor sources
        S_T2 = -hdot exp(-kappa) + g Pi  and  S_E = sqrt(6) g Pi
        (hdot in conformal time, g the visibility function).

        Parameters:
        -----------
        lna : array
            Logarithm of scale factor grid
        modes : array, shape (Ny, Nlna, Nk)
            Tensor perturbation evolution results
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        TensorSourceTable
        """
        BG, params = args
        NF = self.num_F

        delta_g = modes[0]
        shear_g = modes[2]
        F4 = modes[4]
        G0 = modes[NF]
        G2 = modes[NF + 2]
        G4 = modes[NF + 4]
        hdot = modes[-1]

        P2 = -1. / SQRT6 * (
            1. / 10. * delta_g
            + 2. / 7. * shear_g
            + 3. / 70. * F4
            - 3. / 5. * G0
            + 6. / 7. * G2
            - 3. / 70. * G4
        )

        g = vmap(BG.visibility, in_axes=[0, None])(lna, params)[:, None]
        expmkappa = vmap(BG.expmkappa)(lna)[:, None]

        source_T2 = -hdot * expmkappa + g * P2
        source_E = SQRT6 * g * P2

        return TensorSourceTable(self.k_axis_tensor, lna, source_T2, source_E)


class TensorSpectrumSolver(eqx.Module):
    """
    Tensor CMB angular power spectrum computation.

    Integrates the tensor source functions against the flat-space tensor
    radial functions (CLASS transfer.c):
        T : sqrt(3/8 (l+2)(l+1)l(l-1)) j_l(x)/x^2
        E : 1/4 [ j_l'' + 4 j_l'/x - (1 - 2/x^2) j_l ]
        B : 1/2 [ j_l' + 2 j_l/x ]
    with x = k (tau0 - tau), and assembles
        Cl^XY = 4 pi int dk/k P_h(k) Delta_X Delta_Y,
        P_h(k) = r A_s (k/k_pivot)^{n_t}.

    The tensor spectra are computed up to l_tensor_max and zero above it
    (CLASS convention), on the same output ell grid as the scalar solver
    so they can be summed with the scalar spectra before lensing.

    Attributes:
    -----------
    out_ells : jnp.array
        Output multipole grid (the scalar solver's lensing_ells)
    dense_ells : jnp.array
        Multipoles ellmin..l_tensor_max on which the spline is evaluated
    tensor_ells_indices : jnp.array
        Indices into bessel_l_tab covering ellmin..l_tensor_max
    k_axis_transfer : jnp.array
        Wavenumber grid for the transfer integration (units: Mpc^{-1})
    k_pivot : float
        Pivot scale for the primordial spectra (units: Mpc^{-1})

    Methods:
    --------
    primordial_tensor_spectrum : Compute dimensionless P_h(k)
    get_Cl : Compute tensor (TT, TE, EE, BB) on the output ell grid
    Cl_one_ell : Compute tensor spectra for single ell
    """

    out_ells            : jnp.array
    dense_ells          : jnp.array
    tensor_ells_indices : jnp.array
    k_axis_transfer     : jnp.array
    k_pivot             : float = 0.05

    def __init__(self, ellmin, l_tensor_max, out_ells, k_axis_transfer,
                 k_pivot=0.05):
        """
        Initialize tensor spectrum solver.

        Parameters:
        -----------
        ellmin : int
            Minimum multipole
        l_tensor_max : int
            Maximum multipole with tensor contributions (default CLASS: 500)
        out_ells : array
            Output multipole grid to align with (scalar lensing_ells)
        k_axis_transfer : array
            Tensor transfer-integration k grid (units: Mpc^{-1})
        k_pivot : float, optional
            Pivot scale for primordial spectrum (units: Mpc^{-1})
        """
        self.out_ells = out_ells
        self.dense_ells = jnp.arange(ellmin, l_tensor_max + 1)

        ell_idx_min = jnp.where(bessel_l_tab <= ellmin)[0][-1]
        ell_idx_max = jnp.where(bessel_l_tab >= l_tensor_max)[0][0]
        self.tensor_ells_indices = jnp.arange(ell_idx_min, ell_idx_max + 1)

        self.k_axis_transfer = k_axis_transfer
        self.k_pivot = k_pivot

    def primordial_tensor_spectrum(self, k, params):
        """
        Compute dimensionless primordial tensor power spectrum.

        Parameters:
        -----------
        k : float or array
            Wavenumber (units: Mpc^{-1})
        params : dict
            Dictionary of input and derived parameters (uses r, n_t, A_s)

        Returns:
        --------
        float or array
            P_h(k) = r A_s (k/k_pivot)^{n_t}, dimensionless
        """
        return params['r'] * params['A_s'] * (k / self.k_pivot)**params['n_t']

    def get_Cl(self, TPT, BG, params):
        """
        Compute tensor angular power spectra on the output ell grid.

        Parameters:
        -----------
        TPT : TensorSourceTable
            Tensor source function table
        BG : background.Background
            Background cosmology module
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        tuple
            (ClTT, ClTE, ClEE, ClBB) tensor spectra on out_ells, zero
            above l_tensor_max
        """
        tt_raw, te_raw, ee_raw, bb_raw = vmap(
            self.Cl_one_ell, in_axes=(0, None, None, None)
        )(self.tensor_ells_indices, TPT, BG, params)

        node_ells = bessel_l_tab[self.tensor_ells_indices]
        pad = self.out_ells.shape[0] - self.dense_ells.shape[0]

        def to_out_grid(raw):
            dense = CubicSpline(node_ells, raw, check=False)(self.dense_ells)
            return jnp.concatenate((dense, jnp.zeros(pad)))

        return (
            to_out_grid(tt_raw),
            to_out_grid(te_raw),
            to_out_grid(ee_raw),
            to_out_grid(bb_raw),
        )

    def Cl_one_ell(self, idx, TPT, BG, params):
        """
        Compute tensor angular power spectra for single multipole.

        Mirrors the scalar SpectrumSolver.Cl_one_ell structure: interpolate
        sources onto the transfer k grid, accumulate the conformal-time
        integral with a rolling lax.scan over lna, then integrate the
        transfer functions against the primordial tensor spectrum over k.

        Parameters:
        -----------
        idx : int
            Index into bessel_l_tab for multipole ell
        TPT : TensorSourceTable
            Tensor source function table
        BG : background.Background
            Background cosmology module
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        tuple
            (C_l^TT, C_l^TE, C_l^EE, C_l^BB) tensor contributions
        """
        l = bessel_l_tab[idx]
        k_axis = self.k_axis_transfer
        lna_axis = TPT.lna[:-1]
        delta_lna = TPT.lna[-1] - TPT.lna[-2]

        tau0 = BG.tau0
        tau = BG.tau(lna_axis)
        aH = BG.aH(lna_axis, params)

        # Interpolate sources onto the transfer k grid, (Nlna, Nk)
        interp_column = lambda col: CubicSpline(
            jnp.log10(TPT.k), col, check=False)(jnp.log10(k_axis))
        sourceT2 = vmap(interp_column, in_axes=0, out_axes=0)(
            TPT.source_T2[:-1, :])
        sourceE = vmap(interp_column, in_axes=0, out_axes=0)(
            TPT.source_E[:-1, :])

        # Pre-slice bessel-table columns (same pattern as the scalar solver)
        x0_min = xphi0_tab[0, idx]
        x0_max = xphi0_tab[-1, idx]
        x1_min = xphi1_tab[0, idx]
        x1_max = xphi1_tab[-1, idx]
        x2_min = xphi2_tab[0, idx]
        x2_max = xphi2_tab[-1, idx]
        col_phi0_l = phi0_tab[:, idx]
        col_phi1_l = phi1_tab[:, idx]
        col_phi2_l = phi2_tab[:, idx]
        ell_T_factor = jnp.sqrt(3. / 8. * (l + 2) * (l + 1) * l * (l - 1))

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
                    l / x_safe * j(l, x_safe) - j(l + 1, x_safe),
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
                    ((3 * l * (l - 1) - 2 * x_safe**2) * j(l, x_safe)
                     + 6 * x_safe * j(l + 1, x_safe)) / 2 / x_safe**2,
                    tools.fast_interp(x, x2_min, x2_max, col_phi2_l)
                )
            )

        Nlna = lna_axis.shape[0]
        weights = jnp.full((Nlna,), delta_lna, dtype=sourceT2.dtype)
        weights = weights.at[0].set(0.5 * delta_lna)
        zero_k = jnp.zeros(k_axis.shape, dtype=sourceT2.dtype)

        def scan_step(carry, xs_l):
            acc_T, acc_E, acc_B = carry
            sT2_l, sE_l, aH_l, tau_l, w_l = xs_l
            x = (tau0 - tau_l) * k_axis
            p0 = phi0_local(x)
            p1 = phi1_local(x)
            p2 = phi2_local(x)
            jpp = (2. * p2 - p0) / 3.  # j_l'' from phi2 = (3 j'' + j)/2
            radT = ell_T_factor * p0 / x**2
            radE = 0.25 * (jpp + 4. * p1 / x - (1. - 2. / x**2) * p0)
            radB = 0.5 * (p1 + 2. * p0 / x)
            inv_aH = 1.0 / aH_l
            acc_T = acc_T + w_l * sT2_l * inv_aH * radT
            acc_E = acc_E + w_l * sE_l * inv_aH * radE
            acc_B = acc_B + w_l * sE_l * inv_aH * radB
            return (acc_T, acc_E, acc_B), None

        init = (zero_k, zero_k, zero_k)
        xs = (sourceT2, sourceE, aH, tau, weights)
        (transferT, transferE, transferB), _ = lax.scan(
            jax.checkpoint(scan_step), init, xs
        )

        Ph_over_k = 4. * jnp.pi * self.primordial_tensor_spectrum(
            k_axis, params) / k_axis

        return (
            jnp.trapezoid(Ph_over_k * transferT**2, k_axis),
            jnp.trapezoid(Ph_over_k * transferT * transferE, k_axis),
            jnp.trapezoid(Ph_over_k * transferE**2, k_axis),
            jnp.trapezoid(Ph_over_k * transferB**2, k_axis),
        )
