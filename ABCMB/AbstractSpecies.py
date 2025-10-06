import abc
from jax import config, lax
import jax.numpy as jnp
import equinox as eqx
from . import constants as cnst

config.update("jax_enable_x64", True)

### ABSTRACT BASE CLASSES AND INTERFACES ###

class AbstractFluid(eqx.Module, strict=True):
    """
    Abstract base class for fluid species in cosmological simulations.

    Defines an interface for computing fluid thermodynamic properties.

    Methods:
    --------
    rho : Compute energy density (units: eV cm^{-3})
    P : Compute pressure (units: eV cm^{-3})
    cs2 : Compute sound speed squared (units: dimensionless)
    w : Compute equation of state parameter (units: dimensionless)
    """

    @abc.abstractmethod
    def rho(self, lna, BG):
        """
        Compute energy density.

        Calculates the energy density of the fluid species at a given
        cosmological epoch using the logarithm of the scale factor.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Energy density (units: eV cm^{-3})
        """
        raise NotImplementedError("Fluid species must implement an energy density function.")

    @abc.abstractmethod
    def P(self, lna, BG):
        """
        Compute pressure.

        Calculates the pressure of the fluid species at a given
        cosmological epoch using the logarithm of the scale factor.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Pressure (units: eV cm^{-3})
        """
        raise NotImplementedError("Fluid species must implement a pressure function.")

    @abc.abstractmethod
    def cs2(self, lna, BG):
        """
        Compute sound speed squared.

        Calculates the squared sound speed of the fluid species at a given
        cosmological epoch using the logarithm of the scale factor.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Sound speed squared (units: dimensionless)
        """
        raise NotImplementedError("Fluid species must implement a sound speed squared.")

    def w(self, lna, BG):
        """
        Compute equation of state parameter.

        Calculates the ratio of pressure to energy density, representing
        the equation of state for the fluid species.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Equation of state parameter (units: dimensionless)
        """
        return self.P(lna, BG)/self.rho(lna, BG)

class AbstractPerturbedFluid(AbstractFluid, strict=True):
    """
    Abstract base class for fluid species with perturbations.

    Defines methods for computing perturbation mode properties
    used in this code.

    Methods:
    --------
    y_ini : Compute initial perturbation mode conditions (units: dimensionless)
    y_prime : Compute perturbation mode time derivatives (units: dimensionless)
    rho_delta : Compute density perturbation (units: eV cm^{-3})
    rho_plus_P_theta : Compute velocity perturbation (units: eV cm^{-3})
    rho_plus_P_sigma : Compute shear perturbation (units: eV cm^{-3})
    """
    # Some abstract fields that won't be instanced until a child class calls __init__
    # All subclasses must have these fields
    # Declared abstract for now since some methods still reference these fields.
    delta_idx     : eqx.AbstractVar[int]
    num_ell_modes : eqx.AbstractVar[int]

    @abc.abstractmethod
    def y_ini(self, k, tau_ini, om, BG):
        """
        Compute initial conditions for perturbation modes.

        Calculates the initial state of perturbation modes at early cosmological times.

        Parameters:
        ------------
        k : float
            Wavenumber (units: Mpc^{-1})
        tau_ini : float
            Initial conformal time (units: Mpc)
        om : float
            Matter density parameter
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        array
            Initial perturbation mode values (units: dimensionless)
        """
        raise NotImplementedError("Fluid species must implement the initial conditions of their perturbation modes.")

    @abc.abstractmethod
    def y_prime(self, k, lna, metric_h_prime, metric_eta_prime, y, BG):
        """
        Compute time derivatives of perturbation modes.

        Calculates how perturbation modes evolve with cosmological time.

        Parameters:
        ------------
        k : float
            Wavenumber (units: Mpc^{-1})
        lna : float
            Logarithm of scale factor
        metric_h_prime : float
            Derivative of metric h
        metric_eta_prime : float
            Derivative of metric eta
        y : array
            Current perturbation mode values
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        array
            Time derivatives of perturbation modes (units: dimensionless)
        """
        raise NotImplementedError("Fluid species must implement a perturbation derivative function.")

    @abc.abstractmethod
    def rho_delta(self, lna, y, BG):
        """
        Compute density perturbation.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Density perturbation (units: eV cm^{-3})
        """
        raise NotImplementedError("Fluid species must implement a perturbation derivative function.")

    @abc.abstractmethod
    def rho_plus_P_theta(self, lna, y, BG):
        """
        Compute velocity perturbation.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Velocity perturbation (units: eV cm^{-3})
        """
        raise NotImplementedError("Fluid species must implement a perturbation derivative function.")

    @abc.abstractmethod
    def rho_plus_P_sigma(self, lna, y, BG):
        """
        Compute shear perturbation.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Shear perturbation (units: eV cm^{-3})
        """
        raise NotImplementedError("Fluid species must implement a perturbation derivative function.")

class AbstractStandardPerturbedFluid(AbstractPerturbedFluid, strict=True):
    """
    Standard implementation of perturbation methods for fluid species.

    Provides default computations for perturbation-related methods
    used in this code.

    Methods:
    --------
    rho_delta : Compute standard density perturbation (units: eV cm^{-3})
    rho_plus_P_theta : Compute standard velocity perturbation (units: eV cm^{-3})
    rho_plus_P_sigma : Compute standard shear perturbation (units: eV cm^{-3})
    """
    # Called by diffrax, child classes should never override. Okay to implement here.
    def rho_delta(self, lna, y, BG):
        """
        Compute density perturbation.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Density perturbation (units: eV cm^{-3})
        """
        return self.rho(lna, BG) * y[self.delta_idx]

    def rho_plus_P_theta(self, lna, y, BG):
        """
        Compute velocity perturbation.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Velocity perturbation (units: eV cm^{-3})
        """
        return jnp.where(
            self.num_ell_modes > 1,
            (self.rho(lna, BG)+self.P(lna, BG)) * y[self.delta_idx+1],
            0.
        )

    def rho_plus_P_sigma(self, lna, y, BG):
        """
        Compute shear perturbation.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Shear perturbation (units: eV cm^{-3})
        """
        return jnp.where(
            self.num_ell_modes > 2,
            (self.rho(lna, BG)+self.P(lna, BG)) * y[self.delta_idx+2],
            0.
        )

### BEGINNING OF CONCRETE CLASSES ###

class DarkEnergy(AbstractFluid, strict=True):
    """
    Dark energy fluid species implementation.

    Represents a constant energy density fluid with negative pressure.

    Methods:
    --------
    rho : Compute dark energy density (units: eV cm^{-3})
    P : Compute dark energy pressure (units: eV cm^{-3})
    cs2 : Compute sound speed squared (units: dimensionless)
    """
    def rho(self, lna, BG):
        """
        Compute dark energy density.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Dark energy density (units: eV cm^{-3})
        """
        return BG.params['omega_Lambda'] * (3.*cnst.H0_over_h**2/8./jnp.pi/cnst.G)
    
    def P(self, lna, BG):
        """
        Compute dark energy pressure.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Dark energy pressure (units: eV cm^{-3})
        """
        return -self.rho(lna, BG)

    def cs2(self, lna, BG):
        """
        Compute sound speed squared.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Sound speed squared (units: dimensionless)
        """
        return 0.

class ColdDarkMatter(AbstractStandardPerturbedFluid, strict=True):
    """
    Cold dark matter fluid species implementation.

    Non-relativistic, pressureless dark matter with density
    perturbations but no velocity or shear modes.

    Methods:
    --------
    rho : Compute cold dark matter density (units: eV cm^{-3})
    P : Compute cold dark matter pressure (units: eV cm^{-3})
    cs2 : Compute sound speed squared (units: dimensionless)
    y_ini : Compute initial perturbation conditions (units: dimensionless)
    y_prime : Compute perturbation time derivatives (units: dimensionless)
    """

    delta_idx : int
    num_ell_modes = 1

    def rho(self, lna, BG):
        """
        Compute cold dark matter density.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Cold dark matter density (units: eV cm^{-3})
        """
        return BG.params['omega_cdm'] * (3.*cnst.H0_over_h**2/8./jnp.pi/cnst.G) / jnp.exp(lna)**3

    def P(self, lna, BG):
        """
        Compute cold dark matter pressure.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Cold dark matter pressure (units: eV cm^{-3})

        Notes:
        ------
        Cold dark matter is pressureless, so this always returns zero.
        """
        return 0.

    def cs2(self, lna, BG):
        """
        Compute sound speed squared.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Sound speed squared (units: dimensionless)

        Notes:
        ------
        For cold dark matter, sound speed is always zero due to its
        non-relativistic and pressureless nature.
        """
        return 0.
    
    def y_ini(self, k, tau_ini, om, BG):
        """
        Compute initial conditions for cold dark matter perturbations.

        Parameters:
        ------------
        k : float
            Wavenumber (units: Mpc^{-1})
        tau_ini : float
            Initial conformal time (units: Mpc)
        om : float
            Matter density parameter
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        array
            Initial density perturbation (units: dimensionless)
        """
        delta = -(k*tau_ini)**2/4. * (1.-om*tau_ini/5.)
        return jnp.array([delta])

    def y_prime(self, k, lna, metric_h_prime, metric_eta_prime, y, BG):
        """
        Compute time derivatives of cold dark matter perturbations.

        Parameters:
        ------------
        k : float
            Wavenumber (units: Mpc^{-1})
        lna : float
            Logarithm of scale factor
        metric_h_prime : float
            Derivative of metric h
        metric_eta_prime : float
            Derivative of metric eta
        y : array
            Current perturbation mode values
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        array
            Time derivative of density perturbation (units: dimensionless)
        """
        return jnp.array([-0.5*metric_h_prime])

class MasslessNeutrinos(AbstractStandardPerturbedFluid, strict=True):
    """
    Massless neutrinos fluid species implementation.

    Represents relativistic neutrinos with multiple angular momentum modes.

    Methods:
    --------
    rho : Compute neutrino density (units: eV cm^{-3})
    P : Compute neutrino pressure (units: eV cm^{-3})
    cs2 : Compute sound speed squared (units: dimensionless)
    """
    delta_idx : int
    num_ell_modes : int = eqx.field(default=7, static=True)

    def rho(self, lna, BG):
        """
        Compute neutrino density.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Neutrino density (units: eV cm^{-3})
        """
        return BG.params['omega_nu'] * (3.*cnst.H0_over_h**2/8./jnp.pi/cnst.G) / jnp.exp(lna)**4
    
    def P(self, lna, BG):
        """
        Compute neutrino pressure.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Neutrino pressure (units: eV cm^{-3})
        """
        return self.rho(lna, BG)/3.

    def cs2(self, lna, BG):
        """
        Compute sound speed squared.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Sound speed squared (units: dimensionless)

        Notes:
        ------
        For massless neutrinos, sound speed is always 1/3 due to
        their relativistic equation of state.
        """
        return 1./3.

    def y_ini(self, k, tau_ini, om, BG):
        R_nu = BG.params['R_nu']

        delta = - (k*tau_ini)**2/3. * (1.-om*tau_ini/5.)
        theta = - k*(k*tau_ini)**3/36./(4.*R_nu+15.) \
                * (4.*R_nu+11.+12.-3.*(8.*R_nu**2+50.*R_nu+275.)/20./(2.*R_nu+15.)*tau_ini*om)
        sigma = (k*tau_ini)**2/(45.+12.*R_nu) * 2. * (1.+(4.*R_nu-5.)/4./(2.*R_nu+15.)*tau_ini*om)
        
        # Return the four non-zero ell modes, and all higher ell-modes are zero to start.
        # For the neutrinos we track Fnu_2 = 2*sigma, for better structure within the hierarchy.
        return jnp.concatenate((jnp.array([delta, theta, sigma]), jnp.zeros(self.num_ell_modes-3)))

    def y_prime(self, k, lna, metric_h_prime, metric_eta_prime, y, BG):
        aH    = BG.aH(lna)
        tau   = BG.tau(lna)

        L = jnp.arange(self.num_ell_modes) + self.delta_idx
        F = y[L]
        delta = F[0]
        theta = F[1]
        sigma = F[2]

        # density, velocity, shear perturbations
        delta_prime = -4./3./aH*theta - 2./3.*metric_h_prime
        theta_prime = k**2/aH*(delta/4.-sigma)
        sigma_prime = 4./15./aH*theta - 3./10.*k/aH*F[3] + 2./15.*metric_h_prime + 4./5.*metric_eta_prime
        F3_prime = 1./7. * k/aH * (6.*sigma - 4.*F[4])

        # Rest of the Boltzmann Hierarchy
        lmax = self.num_ell_modes-1
        L = jnp.arange(4, lmax)
        Fl_prime    = 1./(2.*L+1.)*k/aH * (L*F[L-1]-(L+1)*F[L+1])
        Flmax_prime = k/aH*F[lmax-1] - (lmax+1)/aH/tau*F[lmax]

        return jnp.concatenate((jnp.array([delta_prime, theta_prime, sigma_prime, F3_prime]), Fl_prime, jnp.array([Flmax_prime])))

class MassiveNeutrinos(AbstractPerturbedFluid, strict=True):
    """
    Massive neutrinos fluid species implementation.

    Non-relativistic neutrinos with multiple angular momentum modes.

    Methods:
    --------
    rho : Compute massive neutrino density (units: eV cm^{-3})
    P : Compute massive neutrino pressure (units: eV cm^{-3})
    cs2 : Compute sound speed squared (units: dimensionless)
    y_ini : Compute initial perturbation conditions (units: dimensionless)
    y_prime : Compute perturbation time derivatives (units: dimensionless)
    rho_delta : Compute density perturbation (units: eV cm^{-3})
    rho_plus_P_theta : Compute velocity perturbation (units: eV cm^{-3})
    rho_plus_P_sigma : Compute shear perturbation (units: eV cm^{-3})
    """

    delta_idx : int

    num_q_bins : int = eqx.field(static=True)
    num_ells_per_bin : int = eqx.field(static=True)
    num_ell_modes : int = eqx.field(static=True)

    q_3p = jnp.array([0.913201, 3.37517, 7.79184])
    w_3p = jnp.array([0.0687359, 3.31435, 2.29911])
    q_5p = jnp.array([0.583165, 2.0, 4.0, 7.26582, 13.0])
    w_5p = jnp.array([0.0081201, 0.689407, 2.8063, 2.05156, 0.12681])

    def __init__(self, delta_idx, num_q_bins=3, num_ells_per_bin=7):
        self.delta_idx = delta_idx
        self.num_q_bins = num_q_bins
        self.num_ells_per_bin = num_ells_per_bin
        self.num_ell_modes = num_q_bins * num_ells_per_bin

    def rho(self, lna, BG):
        """
        Compute massive neutrino density.

        Parameters:
        ------------
        lna : float or ArrayLike
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float or ArrayLike
            Massive neutrino density (units: eV cm^{-3})
        """

        # Ensure lna is at least 1D for broadcasting
        lna_arr = jnp.atleast_1d(lna)          # shape (N,)
        a = jnp.exp(lna_arr)[:, None]          # shape (N, 1)
        T = BG.params['T_ncdm'] / a             # shape (N, 1)
        x = BG.params['m_ncdm'] / T             # shape (N, 1)

        # q_5p, w_5p are shape (5,) → broadcast with (N, 1)
        integrand = (1. + jnp.exp(-self.q_5p)) / self.q_5p**2 \
                    * jnp.sqrt(self.q_5p**2 + x**2)           # (N, 5)

        # Dot product along last axis with w_5p
        integral = jnp.dot(integrand, self.w_5p)               # (N,)

        rho_val = 4. * T[:, 0]**4 / jnp.pi**2 * integral / cnst.hbar**3 / cnst.c**3

        # Remove extra dimension if original input was scalar
        return jnp.squeeze(rho_val) if jnp.ndim(lna) == 0 else rho_val

    def P(self, lna, BG):
        """
        Compute massive neutrino pressure.

        Parameters:
        ------------
        lna : float or ArrayLike
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float or ArrayLike
            Massive neutrino pressure (units: eV cm^{-3})
        """

        # Ensure lna is at least 1D for broadcasting
        lna_arr = jnp.atleast_1d(lna)          # shape (N,)
        a = jnp.exp(lna_arr)[:, None]          # shape (N, 1)
        T = BG.params['T_ncdm'] / a             # shape (N, 1)
        x = BG.params['m_ncdm'] / T             # shape (N, 1)

        # q_5p, w_5p are shape (5,) → broadcast with (N, 1)
        integrand = (1. + jnp.exp(-self.q_5p)) / jnp.sqrt(self.q_5p**2 + x**2) # (N, 5)

        # Dot product along last axis with w_5p
        integral = jnp.dot(integrand, self.w_5p)               # (N,)

        P_val = 4./3. * T[:, 0]**4 / jnp.pi**2 * integral / cnst.hbar**3 / cnst.c**3

        # Remove extra dimension if original input was scalar
        return jnp.squeeze(P_val) if jnp.ndim(lna) == 0 else P_val

    def cs2(self, lna, BG):
        """
        Compute sound speed squared.

        Parameters:
        ------------
        lna : float or ArrayLike
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float or ArrayLike
            Sound speed squared (units: dimensionless)

        Notes:
        ------
        Uses equation of state parameter w as approximation.
        """
        return self.w(lna, BG) # ZZ : Is this correct?

    def y_ini(self, k, tau_ini, om, BG):
        """
        Compute initial conditions for massive neutrino perturbations.

        Parameters:
        ------------
        k : float
            Wavenumber (units: Mpc^{-1})
        tau_ini : float
            Initial conformal time (units: Mpc)
        om : float
            Matter density parameter
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        array
            Initial perturbation mode values (units: dimensionless)
        """
        res = jnp.zeros(self.num_ell_modes)

        # Initial conditions for massless neutrinos first, needed here.
        R_nu = BG.params['R_nu']

        delta = - (k*tau_ini)**2/3. * (1.-om*tau_ini/5.)
        theta = - k*(k*tau_ini)**3/36./(4.*R_nu+15.) \
                * (4.*R_nu+11.+12.-3.*(8.*R_nu**2+50.*R_nu+275.)/20./(2.*R_nu+15.)*tau_ini*om)
        sigma = (k*tau_ini)**2/(45.+12.*R_nu) * 2. * (1.+(4.*R_nu-5.)/4./(2.*R_nu+15.)*tau_ini*om)

        dlogf0_dlogq = self.q_3p / (1.+jnp.exp(-self.q_3p)) # Log derivative of the fermi-dirac distribution.

        idx_q1 = 0 # Psi0 index of first q
        idx_q2 = idx_q1 + 7 # Psi0 index of second q
        idx_q3 = idx_q2 + 7 # Psi0 index of third q

        for i in range(3):
            q  = self.q_3p[i] # This momentum bin.
            iq = i*self.num_ells_per_bin # Index in diffrax array of this momentum bin.
            # ZZ : Techniclly Psi1 requires epsilon/q = 1/v, but at early times this should be 1. Should check this accuracy!
            first_three = jnp.array([delta/4., theta/3., sigma/2.]) * q / (1.+jnp.exp(-q))
            res = res.at[iq:iq+3].set(first_three)

        return res

    def y_prime(self, k, lna, metric_h_prime, metric_eta_prime, y, BG):
        """
        Compute time derivatives of massive neutrino perturbations.

        Parameters:
        ------------
        k : float
            Wavenumber (units: Mpc^{-1})
        lna : float
            Logarithm of scale factor
        metric_h_prime : float
            Derivative of metric h
        metric_eta_prime : float
            Derivative of metric eta
        y : array
            Current perturbation mode values
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        array
            Time derivatives of perturbation modes (units: dimensionless)
        """
        res = jnp.zeros(self.num_ell_modes)

        a = jnp.exp(lna)
        T = BG.params['T_ncdm'] / a
        x = BG.params['m_ncdm'] / T
        aH  = BG.aH(lna)
        tau = BG.tau(lna)

        # Iterate through momentum bins
        for i in range(3):
            q = self.q_3p[i]
            epsilon = jnp.sqrt(q**2 + x**2)
            dlnf0_dlnq = -q / (q+jnp.exp(-q))

            # NOTE: The entries are [Psi0, k * Psi1, Psi2, ...]. If accessing Psi1 make sure to divide out k
            L = jnp.arange(self.num_ells_per_bin) + self.delta_idx + i*self.num_ells_per_bin
            Psi = y[L]

            Psi0_prime = -q*k/epsilon/aH*Psi[1] + metric_h_prime/6. * dlnf0_dlnq
            kPsi1_prime = q*k**2/3./epsilon/aH * (Psi[0] - 2.*Psi[2])
            Psi2_prime = q*k/5./epsilon/aH * (2.*Psi[1]/k - 3.*Psi[3]) - (metric_h_prime/15. + 2.*metric_eta_prime/5.) * dlnf0_dlnq
            
            # Intermediate hierarchy, 3<=L<lmax
            lmax = self.num_ells_per_bin - 1
            L_inter = jnp.arange(3, lmax) # Doesn't include lmax.
            Psi_inter_prime = q*k/epsilon/aH/(2*L_inter+1) * (L_inter*Psi[L_inter-1] - (L_inter+1)*Psi[L_inter+1])

            # lmax mode
            Psi_lmax_prime = q*k/aH/epsilon*Psi[lmax-1] - (lmax+1)/aH/tau*Psi[lmax]

            # Putting it all together
            subres = jnp.concatenate((jnp.array([Psi0_prime, kPsi1_prime, Psi2_prime]), Psi_inter_prime, jnp.array([Psi_lmax_prime])))
            res = res.at[i*self.num_ells_per_bin:(i+1)*self.num_ells_per_bin].set(subres)

        return res

    def rho_delta(self, lna, y, BG):
        """
        Compute massive neutrino density perturbation.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Density perturbation (units: eV cm^{-3})
        """
        a = jnp.exp(lna)
        T = BG.params['T_ncdm'] / a  # (N,)
        x = BG.params['m_ncdm'] / T  # (N,)

        res = 0.
        for i in range(self.num_q_bins):
            q = self.q_3p[i]
            w = self.w_3p[i]
            epsilon = jnp.sqrt(q**2 + x**2)
            Psi0 = y[self.delta_idx + i*self.num_ells_per_bin]

            res += w*(1.+jnp.exp(-q))*epsilon/q**2 * Psi0
        return res * 4./jnp.pi**2 * T**4 / cnst.hbar**3 / cnst.c**3

    def rho_plus_P_theta(self, lna, y, BG):
        """
        Compute massive neutrino velocity perturbation.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Velocity perturbation (units: eV cm^{-3})
        """
        a = jnp.exp(lna)
        T = BG.params['T_ncdm'] / a  # (N,)
        x = BG.params['m_ncdm'] / T  # (N,)

        res = 0.
        for i in range(self.num_q_bins):
            q = self.q_3p[i]
            w = self.w_3p[i]
            kPsi1 = y[self.delta_idx+1 + i*self.num_ells_per_bin]

            res += w*(1.+jnp.exp(-q))/q * kPsi1
        return res * 4./jnp.pi**2 * T**4 / cnst.hbar**3 / cnst.c**3

    def rho_plus_P_sigma(self, lna, y, BG):
        """
        Compute massive neutrino shear perturbation.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Shear perturbation (units: eV cm^{-3})
        """
        a = jnp.exp(lna)
        T = BG.params['T_ncdm'] / a  # (N,)
        x = BG.params['m_ncdm'] / T  # (N,)

        res = 0.
        for i in range(self.num_q_bins):
            q = self.q_3p[i]
            w = self.w_3p[i]
            epsilon = jnp.sqrt(q**2 + x**2)
            Psi2 = y[self.delta_idx+2 + i*self.num_ells_per_bin]

            res += w*(1.+jnp.exp(-q))/epsilon * Psi2
        return res * 8./3./jnp.pi**2 * T**4 / cnst.hbar**3 / cnst.c**3


class Baryon(AbstractStandardPerturbedFluid, strict=True):
    """
    Baryon fluid species implementation.

    Non-relativistic baryons with density and velocity perturbations.

    Methods:
    --------
    rho : Compute baryon density (units: eV cm^{-3})
    P : Compute baryon pressure (units: eV cm^{-3})
    cs2 : Compute sound speed squared (units: dimensionless)
    mean_mass : Compute mean baryon mass (units: eV)
    y_ini : Compute initial perturbation conditions (units: dimensionless)
    y_prime : Compute perturbation time derivatives (units: dimensionless)
    """
    delta_idx : int
    #coupled_delta_idx : int # Index of coupled photon
    photon : AbstractPerturbedFluid
    num_ell_modes = 2

    def rho(self, lna, BG):
        """
        Compute baryon density.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Baryon density (units: eV cm^{-3})
        """
        return BG.params['omega_b'] * (3.*cnst.H0_over_h**2/8./jnp.pi/cnst.G) / jnp.exp(lna)**3

    def P(self, lna, BG):
        """
        Compute baryon pressure.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Baryon pressure (units: eV cm^{-3})

        Notes:
        ------
        Baryon pressure is neglected, standard practice for SM baryons.
        """
        return 0.

    def cs2(self, lna, BG):
        """
        Compute sound speed squared.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Sound speed squared (units: dimensionless)

        Notes:
        ------
        Adiabatic sound speed squared, taken from M&B Eq. (68).
        Although we can neglect the pressure, this term is important for perturbation growth
        during recombination. During reionization this cs2 is negative. This is not physical
        but it should not matter for cosmology.
        """
        Tm = BG.Tm(lna) # Baryon temp
        Tg = BG.TCMB(lna) # Photon temp
        mu = self.mean_mass(lna, BG)
        R = 4.*self.photon.rho(lna, BG)/3./self.rho(lna, BG)

        return Tm/mu * (5./3. - 2./3.*mu*R/cnst.me/BG.aH(lna)/BG.tau_c(lna) * (Tg/Tm - 1.))

    def mean_mass(self, lna, BG):
        """
        Compute mean baryon mass at given redshift.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Mean baryon mass (units: eV)

        Notes:
        ------
        Defined to be mu = rho_b / n_b = rho_b / (nH + nHe + ne)
        """
        denom = (1.+BG.xe(lna))*(1.-BG.params['YHe']) + cnst.mH / cnst.mHe * BG.params['YHe']
        return cnst.mH / denom

    def y_ini(self, k, tau_ini, om, BG):
        """
        Compute initial conditions for baryon perturbations.

        Parameters:
        ------------
        k : float
            Wavenumber (units: Mpc^{-1})
        tau_ini : float
            Initial conformal time (units: Mpc)
        om : float
            Matter density parameter
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        array
            Initial perturbation mode values (units: dimensionless)
        """
        delta = -(k*tau_ini)**2/4. * (1.-om*tau_ini/5.)
        theta = - k**4 * tau_ini**3/36. * (1.-3.*(1.+5.*BG.params['R_b']-BG.params['R_nu'])/20./(1.-BG.params['R_nu'])*om*tau_ini)
        return jnp.array([delta, theta])

    def y_prime(self, k, lna, metric_h_prime, metric_eta_prime, y, BG):
        """
        Compute time derivatives of baryon perturbations.

        Parameters:
        ------------
        k : float
            Wavenumber (units: Mpc^{-1})
        lna : float
            Logarithm of scale factor
        metric_h_prime : float
            Derivative of metric h
        metric_eta_prime : float
            Derivative of metric eta
        y : array
            Current perturbation mode values
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        array
            Time derivatives of perturbation modes (units: dimensionless)
        """
        aH = BG.aH(lna)
        cs2 = self.cs2(lna, BG)
        R = 4.*self.photon.rho(lna, BG)/3./self.rho(lna, BG)
        tau_c = BG.tau_c(lna)

        delta = y[self.delta_idx]
        theta = y[self.delta_idx+1]
        theta_g = y[self.photon.delta_idx+1]
        delta_prime = -theta/aH-metric_h_prime/2.
        theta_prime = -theta + cs2*k**2*delta/aH + R/tau_c/aH*(theta_g-theta)
        
        return jnp.array([delta_prime, theta_prime])

class Photon(AbstractStandardPerturbedFluid, strict=True):
    """
    Photon fluid species implementation.

    Relativistic photons with temperature and polarization Boltzmann hierarchies.

    Methods:
    --------
    rho : Compute photon density (units: eV cm^{-3})
    P : Compute photon pressure (units: eV cm^{-3})
    cs2 : Compute sound speed squared (units: dimensionless)
    y_ini : Compute initial perturbation conditions (units: dimensionless)
    y_prime : Compute perturbation time derivatives (units: dimensionless)
    """
    delta_idx : int
    baryon : AbstractPerturbedFluid 
    num_F_ell_modes : int = eqx.field(static=True)
    num_G_ell_modes : int = eqx.field(static=True)
    num_ell_modes : int = eqx.field(static=True)

    def __init__(self, delta_idx, baryon, num_F_ell_modes=7, num_G_ell_modes=7):
        self.delta_idx = delta_idx
        self.baryon = baryon
        self.num_F_ell_modes = num_F_ell_modes
        self.num_G_ell_modes = num_G_ell_modes
        self.num_ell_modes = num_F_ell_modes + num_G_ell_modes

    def rho(self, lna, BG):
        """
        Compute photon density.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Photon density (units: eV cm^{-3})
        """
        return BG.params['omega_g'] * (3.*cnst.H0_over_h**2/8./jnp.pi/cnst.G) / jnp.exp(lna)**4

    def P(self, lna, BG):
        """
        Compute photon pressure.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Photon pressure (units: eV cm^{-3})
        """
        return self.rho(lna, BG)/3.

    def cs2(self, lna, BG):
        """
        Compute sound speed squared.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Sound speed squared (units: dimensionless)

        Notes:
        ------
        For photons, sound speed is always 1/3 due to their relativistic equation of state.
        """
        return 1./3.

    def y_ini(self, k, tau_ini, om, BG):
        """
        Compute initial conditions for photon perturbations.

        Parameters:
        ------------
        k : float
            Wavenumber (units: Mpc^{-1})
        tau_ini : float
            Initial conformal time (units: Mpc)
        om : float
            Matter density parameter
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        array
            Initial perturbation mode values (units: dimensionless)
        """
        delta = - (k*tau_ini)**2/3. * (1.-om*tau_ini/5.)
        theta = - k**4 * tau_ini**3/36. * (1.-3.*(1.+5.*BG.params['R_b']-BG.params['R_nu'])/20./(1.-BG.params['R_nu'])*om*tau_ini)
        return jnp.concatenate((jnp.array([delta, theta]), jnp.zeros(self.num_ell_modes - 2)))

    def y_prime(self, k, lna, metric_h_prime, metric_eta_prime, y, BG):
        """
        Compute time derivatives of photon perturbations.

        Parameters:
        ------------
        k : float
            Wavenumber (units: Mpc^{-1})
        lna : float
            Logarithm of scale factor
        metric_h_prime : float
            Derivative of metric h
        metric_eta_prime : float
            Derivative of metric eta
        y : array
            Current perturbation mode values
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        array
            Time derivatives of perturbation modes (units: dimensionless)
        """
        aH    = BG.aH(lna)
        tau_c = BG.tau_c(lna)
        tau   = BG.tau(lna)

        Flmax = self.num_F_ell_modes-1
        Glmax = self.num_G_ell_modes-1
        F = lax.dynamic_slice(y, (self.delta_idx,), (self.num_F_ell_modes,))
        G = lax.dynamic_slice(y, (self.delta_idx+self.num_F_ell_modes,), (self.num_G_ell_modes,))
        delta = F[0]
        theta = F[1]
        sigma = F[2]
        theta_b = y[self.baryon.delta_idx+1]

        delta_prime = -4./3./aH*theta - 2./3.*metric_h_prime
        theta_prime = k**2/aH*(delta/4.-sigma) + (theta_b-theta)/aH/tau_c
        sigma_prime = 4./15./aH*theta - 3./10.*k/aH*F[3] + 2./15.*metric_h_prime + 4./5.*metric_eta_prime - 9./10./aH/tau_c*sigma + (G[0]+G[2])/20./aH/tau_c
        F3_prime    = k/7./aH * (6.*sigma - 4.*F[4]) - F[3]/aH/tau_c

        # Temperature Boltzmann Hierarchy
        L = jnp.arange(4, Flmax) # Excludes the lmax mode
        Fl_prime    = 1./(2.*L+1.)*k/aH * (L*F[L-1]-(L+1)*F[L+1]) - F[L]/aH/tau_c
        Flmax_prime = k/aH*F[Flmax-1] - (Flmax+1)/aH/tau*F[Flmax] - F[Flmax]/aH/tau_c

        # Polarization Boltzmann Hierarchy
        L = jnp.arange(0, Glmax) # Excludes the lmax mode
        Gl_prime    = 1./(2.*L+1.)*k/aH * (L*G[L-1]-(L+1)*G[L+1]) - G[L]/aH/tau_c \
                    + (2.*sigma+G[0]+G[2])/2./aH/tau_c * jnp.concatenate((jnp.array([1., 0., 0.2]), jnp.zeros(Glmax-3)))

        Glmax_prime = k/aH*G[Glmax-1] - (Glmax+1)/aH/tau*G[Glmax] - G[Glmax]/aH/tau_c
        return jnp.concatenate((jnp.array([delta_prime, theta_prime, sigma_prime, F3_prime]), Fl_prime, jnp.array([Flmax_prime]), Gl_prime, jnp.array([Glmax_prime])))