#import abc
from jax import config, lax, vmap
import jax.numpy as jnp
import equinox as eqx
from . import constants as cnst
from .ABCMBTools import cot_K

config.update("jax_enable_x64", True)


def curvature_s_l(l, k, K):
    """
    Curved-space free-streaming coefficients s_l = sqrt(1 - K(l^2-1)/k^2).

    Dresses the l <-> l+1 couplings of every Boltzmann hierarchy in non-flat
    FLRW space (Lesgourgues & Tram, arXiv:1305.3261; CLASS perturbations.c
    s_l array). s_1 = 1 and all s_l = 1 in the flat limit, so the flat
    equations are recovered exactly at K = 0. Clipped at zero (with an
    AD-safe sqrt argument): for closed universes modes only support l <= nu-1,
    and the clip terminates the hierarchy physically.

    Parameters
    ----------
    l : array
        Multipole indices.
    k : float
        Wavenumber eigenvalue label, Mpc^-1.
    K : float
        Curvature constant, Mpc^-2.

    Returns
    -------
    array
        s_l, same shape as l.
    """
    arg = 1. - K*(l*l - 1.)/k**2
    return jnp.where(arg > 0., jnp.sqrt(jnp.clip(arg, 1e-30, None)), 0.)


### ABSTRACT BASE CLASSES AND INTERFACES ###

class Fluid(eqx.Module):
    """
    Base class for fluid species.

    Defines fluid properties.

    Fields:
    -------
    first_idx : int
        Default = 0
        Position of the first perturbation equation in the Diffrax vector. For most fluids this is the density perturbation mode "delta".
    num_equations : int 
        Default = 0
        Number of equations that need to be simultaneously evolved in the perturbations module. 
    name : str
        Default = ""
        Name of the fluid, used to find fluid and refer to it later in the computation using species_dict["name"].
    is_matter : bool
        Default = False
        Whether the fluid is non-relativistic today and contributes towards the total matter power spectrum. 

    Methods:
    --------
        rho : Compute energy density (units: eV cm^{-3})
        P   : Compute pressure (units: eV cm^{-3})
        w   : Compute equation of state parameter (units: dimensionless)
        y_ini   : Adiabatic initial conditions, in synchronous gauge
        y_prime : Perturbation derivatives, in synchronous gauge
        rho_delta        : Perturbed density function δρ (units: eV cm^{-3})
        rho_plus_P_theta : Velocity perturbation  (units: eV cm^{-3} Mpc^{-1})
        rho_plus_P_sigma : Compute standard shear perturbation (units: eV cm^{-3})
    """

    first_idx     : int = eqx.field(default=0, static=True)
    num_equations : int = eqx.field(default=0, static=True)
    name          : str = eqx.field(default="", static=True)
    is_matter     : bool = eqx.field(default=False, static=True) # Does the fluid contribute towards matter overdensity today.

    def __init__(self, first_idx, specs):
        self.first_idx = first_idx
        self.name = ""
        self.is_matter = False

    def rho(self, lna, args):
        """
        Compute energy density.

        Calculates the energy density of the fluid species at a given
        cosmological epoch using the logarithm of the scale factor.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Energy density (units: eV cm^{-3})
        """
        raise NotImplementedError("Fluid species must implement an energy density function.")

    def P(self, lna, args):
        """
        Compute pressure.

        Calculates the pressure of the fluid species at a given
        cosmological epoch using the logarithm of the scale factor.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Pressure (units: eV cm^{-3})
        """
        raise NotImplementedError("Fluid species must implement a pressure function.")

    def w(self, lna, args):
        """
        Compute equation of state parameter.

        Calculates the ratio of pressure to energy density, representing
        the equation of state for the fluid species.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Equation of state parameter (units: dimensionless)
        """
        return self.P(lna, args)/self.rho(lna, args)

    def y_ini(self, k, tau_ini, args):
        """
        Compute initial conditions for perturbation modes.

        Calculates the initial state of perturbation modes at early cosmological times.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        tau_ini : float
            Initial conformal time (units: Mpc)
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        array
            Initial perturbation mode values
        """
        raise NotImplementedError("Fluid species must implement the initial conditions of their perturbation modes.")

    def y_prime(self, k, lna, metric_h_prime, metric_eta_prime, y, args):
        """
        Compute time derivatives of perturbation modes.

        Calculates how perturbation modes evolve with cosmological time.

        Parameters:
        -----------
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
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        array
            Time derivatives of perturbation modes
        """
        raise NotImplementedError("Fluid species must implement a perturbation derivative function.")

    def rho_delta(self, lna, y, args):
        """
        Compute density perturbation.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Density perturbation (units: eV cm^{-3})
        """
        raise NotImplementedError("Fluid species must implement a perturbation derivative function.")

    def rho_plus_P_theta(self, lna, y, args):
        """
        Compute velocity perturbation.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Velocity perturbation (units: eV cm^{-3} Mpc^{-1})
        """
        raise NotImplementedError("Fluid species must implement a perturbation derivative function.")

    def rho_plus_P_sigma(self, lna, y, args):
        """
        Compute shear perturbation.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Shear perturbation (units: eV cm^{-3})
        """
        raise NotImplementedError("Fluid species must implement a perturbation derivative function.")

    def output_perturbations(self, lna, modes, args):
        """
        Return named perturbation arrays for storage in PerturbationTable.

        Each concrete species overrides this to select the physically
        meaningful subset of its modes. Species with no perturbations
        (e.g. dark energy) return an empty dict via this base implementation.

        Parameters:
        -----------
        lna : array, shape (Nlna,)
            Logarithm of scale factor grid
        modes : array, shape (Ny, Nlna, Nk)
            Full perturbation state, already transposed
        args : tuple
            (BG, params) — background cosmology and cosmological parameters

        Returns:
        --------
        dict
            {quantity_name: array(Nlna, Nk)}. Empty for background-only species.
        """
        return {}

class StandardFluid(Fluid):
    """
    Standard implementation of perturbation methods for fluid species.

    Provides default computations for perturbation-related methods
    used in this code.

    Methods:
    --------
    rho_delta : Compute standard density perturbation (units: eV cm^{-3})
    rho_plus_P_theta : Compute standard velocity perturbation (units: eV cm^{-3} Mpc^{-1})
    rho_plus_P_sigma : Compute standard shear perturbation (units: eV cm^{-3})
    """

    def __init__(self, first_idx, specs):
        super().__init__(first_idx, specs)

    def get_delta(self, lna, y, args):
        """
        Getter method for density perturbation from perturbation equations vector

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Dimensionless density perturbation (units: None)
        """
        return y[self.first_idx]
    
    def get_theta(self, lna, y, args):
        """
        Getter method for velocity divergence perturbation from perturbation equations vector

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Velocity divergence perturbation (units: 1/Mpc)
        """
        return jnp.where(
            self.num_equations > 1,
            y[self.first_idx+1],
            0
        )
    
    def get_sigma(self, lna, y, args):
        """
        Getter method for shear perturbation from perturbation equations vector

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Dimensionless shear perturbation (units: None)
        """
        return jnp.where(
            self.num_equations > 2,
            y[self.first_idx+2],
            0
        )

    # Called by diffrax, child classes should never override. Okay to implement here.
    def rho_delta(self, lna, y, args):
        """
        Compute energy density perturbation, contribution to metric perturbation evolution. 

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Energy density perturbation (units: eV cm^{-3})
        """
        params = args
        return self.rho(lna, params) * self.get_delta(lna, y, args)

    def rho_plus_P_theta(self, lna, y, args):
        """
        Compute velocity perturbation times the sum of energy density and pressure. {0, i} component
        of the perturbed stress energy tensor.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Velocity perturbation (units: eV cm^{-3} Mpc^{-1})
        """
        params = args
        return (self.rho(lna, params)+self.P(lna, params)) * self.get_theta(lna, y, args)


    def rho_plus_P_sigma(self, lna, y, args):
        """
        Compute shear stress perturbation, needed for CMB

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
           Shear stress perturbation (units: eV cm^{-3})
        """
        params = args
        return (self.rho(lna, params)+self.P(lna, params)) * self.get_sigma(lna, y, args)

class BackgroundFluid(Fluid):
    
    num_equations = 0

    def __init__(self, first_idx, specs):
        super().__init__(first_idx, specs)

    def y_ini(self, k, tau_ini, args):
        """
        Trivial initial condition vector for background.
        """
        return jnp.array([])

    def y_prime(self, k, lna, metric_h_prime, metric_eta_prime, y, args):
        """
        Trivial derivative vector for background. 
        """
        return jnp.array([])

    def rho_delta(self, lna, y, args):
        return 0.

    def rho_plus_P_theta(self, lna, y, args):
        return 0.

    def rho_plus_P_sigma(self, lna, y, args):
        return 0.

### BEGINNING OF CONCRETE CLASSES ###

class DarkEnergy(BackgroundFluid):
    """
    Dark energy fluid species implementation.

    Represents a constant energy density fluid with negative pressure.

    Required input parameters: None.

    Required derived parameters: params['omega_Lambda']

    Methods:
    --------
    rho : Compute dark energy density (units: eV cm^{-3})
    P : Compute dark energy pressure (units: eV cm^{-3})
    """

    name = "DarkEnergy"

    def __init__(self, first_idx, specs):
        super().__init__(first_idx, specs)
        self.name = "DarkEnergy"

    def rho(self, lna, args):
        """
        Compute dark energy density.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Dark energy density (units: eV cm^{-3})
        """
        params = args
        return params['omega_Lambda'] * (3.*cnst.H0_over_h**2/8./jnp.pi/cnst.G)
    
    def P(self, lna, args):
        """
        Compute dark energy pressure.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Dark energy pressure (units: eV cm^{-3})
        """
        params = args
        return -self.rho(lna, params)

class Curvature(BackgroundFluid):
    """
    Spatial-curvature pseudo-fluid.

    Represents the -K c^2/a^2 term of the curved Friedmann equation as a
    background species with rho ~ a^{-2} and w = -1/3, so that H, aH_prime
    and d2adtau2_over_a inherit the curved forms with no changes to the
    background machinery (rho + 3P = 0 for w = -1/3, so the constant -K c^2
    drops out of the (aH)^2 derivative exactly as it should). Carries no
    perturbations: the perturbed Einstein equations and Boltzmann hierarchies
    take curvature through explicit K factors instead.

    Required input parameters: params['omega_k'] (= Omega_k h^2; positive for
    an open universe, CLASS sign convention).

    Methods:
    --------
    rho : Compute curvature pseudo-density (units: eV cm^{-3}; negative for closed)
    P : Compute curvature pseudo-pressure (units: eV cm^{-3})
    """

    name = "Curvature"

    def __init__(self, first_idx, specs):
        super().__init__(first_idx, specs)
        self.name = "Curvature"

    def rho(self, lna, args):
        """
        Compute curvature pseudo-density.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Curvature pseudo-density (units: eV cm^{-3})
        """
        params = args
        return params['omega_k'] * (3.*cnst.H0_over_h**2/8./jnp.pi/cnst.G) / jnp.exp(lna)**2

    def P(self, lna, args):
        """
        Compute curvature pseudo-pressure (w = -1/3).

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Curvature pseudo-pressure (units: eV cm^{-3})
        """
        params = args
        return -self.rho(lna, params)/3.

class ColdDarkMatter(StandardFluid):
    """
    Cold dark matter fluid species implementation.

    Non-relativistic, pressureless dark matter with density
    perturbations but no velocity or shear modes.

    Required input parameters: params['omega_cdm'].

    Required derived parameters: params['om'].

    Methods:
    --------
    rho : Compute cold dark matter density (units: eV cm^{-3})
    P : Compute cold dark matter pressure (units: eV cm^{-3})
    y_ini : Compute initial perturbation conditions
    y_prime : Compute perturbation time derivatives
    """

    name = "ColdDarkMatter"
    num_equations = 1  # CDM only receives density perturbation in synchronous gauge.
    is_matter = True

    def __init__(self, first_idx, specs):
        super().__init__(first_idx, specs)
        self.name = "ColdDarkMatter"
        self.is_matter = True

    def rho(self, lna, args):
        """
        Compute cold dark matter density.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Cold dark matter density (units: eV cm^{-3})
        """
        params = args
        return params['omega_cdm'] * (3.*cnst.H0_over_h**2/8./jnp.pi/cnst.G) / jnp.exp(lna)**3

    def P(self, lna, args):
        """
        Compute cold dark matter pressure.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Cold dark matter pressure (units: eV cm^{-3})

        Notes:
        ------
        Cold dark matter is pressureless, so this always returns zero.
        """
        return 0.
    
    def y_ini(self, k, tau_ini, args):
        """
        Compute initial conditions for cold dark matter perturbations.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        tau_ini : float
            Initial conformal time (units: Mpc)
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        array
            Initial density perturbation (units: dimensionless)
        """
        params = args
        s2_squared = 1. - 3.*params['K']/k**2
        delta = -(k*tau_ini)**2/4. * (1.-params["om"]*tau_ini/5.) * s2_squared
        return jnp.array([delta])

    def y_prime(self, k, lna, metric_h_prime, metric_eta_prime, y, args):
        """
        Compute time derivatives of cold dark matter perturbations.

        Parameters:
        -----------
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
        args : tuple
            Background cosmology and cosmological parameters (BG, params) - Note: BG parameter is unused in this implementation

        Returns:
        --------
        array
            Time derivative of density perturbation (units: dimensionless)
        """
        return jnp.array([-0.5*metric_h_prime])

    def output_perturbations(self, lna, modes, args):
        return {"delta": modes[self.first_idx]}

class MasslessNeutrino(StandardFluid):
    """
    Massless neutrinos fluid species implementation.

    Represents relativistic neutrinos with multiple angular momentum modes.

    Required input parameters: params['N_nu_massless'], params['T_nu_massless'], params['TCMB0']

    Required derived parameters: params['R_nu'], params['om']

    Methods:
    --------
    rho : Compute neutrino density (units: eV cm^{-3})
    P : Compute neutrino pressure (units: eV cm^{-3})
    """

    name = "MasslessNeutrino"

    def __init__(self, first_idx, specs):
        super().__init__(first_idx, specs)
        self.name = "MasslessNeutrino"
        self.num_equations = specs["l_max_massless_nu"] + 1

    def rho(self, lna, args):
        """
        Compute neutrino density.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Neutrino density (units: eV cm^{-3})
        """
        params = args

        a = jnp.exp(lna)
        rho = params['N_nu_massless'] * 2. * 7./8. * jnp.pi**2/30. * params['T_nu_massless']**4 * params['TCMB0']**4 / a**4 # eV^4
        rho = rho / (cnst.c * cnst.hbar)**3 # Convert to eV cm^{-3}
        return rho
    
    def P(self, lna, args):
        """
        Compute neutrino pressure.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Neutrino pressure (units: eV cm^{-3})
        """
        params = args
        return self.rho(lna, params)/3.

    def y_ini(self, k, tau_ini, args):
        """
        Compute initial conditions for massless neutrino perturbations.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        tau_ini : float
            Initial conformal time (units: Mpc)
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        array
            Initial perturbation mode values (units: 1/Mpc for theta, else dimensionless)
        """
        params = args
        R_nu = params['R_nu']
        s2_squared = 1. - 3.*params['K']/k**2

        delta = - (k*tau_ini)**2/3. * (1.-params["om"]*tau_ini/5.) * s2_squared
        theta = - k*(k*tau_ini)**3/36./(4.*R_nu+15.) \
                * (4.*R_nu+11.+12.*s2_squared-3.*(8.*R_nu**2+50.*R_nu+275.)/20./(2.*R_nu+15.)*tau_ini*params["om"]) * s2_squared
        sigma = (k*tau_ini)**2/(45.+12.*R_nu) * (3.*s2_squared-1.) * (1.+(4.*R_nu-5.)/4./(2.*R_nu+15.)*tau_ini*params["om"])

        # Return the four non-zero ell modes, and all higher ell-modes are zero to start.
        # For the neutrinos we track Fnu_2 = 2*sigma, for better structure within the hierarchy.
        return jnp.concatenate((jnp.array([delta, theta, sigma]), jnp.zeros(self.num_equations-3)))

    def y_prime(self, k, lna, metric_h_prime, metric_eta_prime, y, args):
        """
        Compute time derivatives of massless neutrino perturbations.

        Parameters:
        -----------
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
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        array
            Time derivatives of perturbation modes (units: 1/Mpc for theta, else dimensionless)
        """
        BG, params, _, _ = args
        aH    = BG.aH(lna, params)
        tau   = BG.tau(lna)

        L = jnp.arange(self.num_equations) + self.first_idx
        F = y[L]
        delta = F[0]
        theta = F[1]
        sigma = F[2]

        # Curved-space free-streaming coefficients and l_max truncation,
        # CLASS perturbations.c ur hierarchy a la arXiv:1305.3261.
        K = params['K']
        s = curvature_s_l(jnp.arange(self.num_equations), k, K)
        cotK_tau = cot_K(tau, K)

        # density, velocity, shear perturbations
        delta_prime = -4./3./aH*theta - 2./3.*metric_h_prime
        theta_prime = k**2/aH*(delta/4.-s[2]**2*sigma)
        sigma_prime = 4./15./aH*theta - 3./10.*k/aH*s[3]/s[2]*F[3] + 2./15.*metric_h_prime + 4./5.*metric_eta_prime
        F3_prime = 1./7. * k/aH * (6.*s[3]*s[2]*sigma - 4.*s[4]*F[4])

        # Rest of the Boltzmann Hierarchy
        lmax = self.num_equations-1
        L = jnp.arange(4, lmax)
        Fl_prime    = 1./(2.*L+1.)*k/aH * (L*s[L]*F[L-1]-(L+1)*s[L+1]*F[L+1])
        Flmax_prime = k/aH*s[lmax]*F[lmax-1] - (lmax+1)*cotK_tau/aH*F[lmax]

        return jnp.concatenate((jnp.array([delta_prime, theta_prime, sigma_prime, F3_prime]), Fl_prime, jnp.array([Flmax_prime])))

    def output_perturbations(self, lna, modes, args):
        return {
            "delta": modes[self.first_idx],
            "theta": modes[self.first_idx + 1],
            "sigma": modes[self.first_idx + 2],
        }

class MassiveNeutrino(Fluid):
    """
    Massive neutrinos fluid species implementation.

    Non-relativistic neutrinos with multiple angular momentum modes.

    Required input parameters: params['N_nu_massive'], params['T_nu_massive'], 
    params['m_nu_massive'], params['TCMB0']

    Required derived parameters: params['R_nu'], params['om']

    Attributes:
    -----------
    num_ells_per_bin : int
        Number of multipole moments per momentum bin for massive neutrino hierarchy

    Methods:
    --------
    rho : Compute massive neutrino density (units: eV cm^{-3})
    P : Compute massive neutrino pressure (units: eV cm^{-3})
    y_ini : Compute initial perturbation conditions 
    y_prime : Compute perturbation time derivatives 
    rho_delta : Compute density perturbation (units: eV cm^{-3})
    rho_plus_P_theta : Compute velocity perturbation (units: eV cm^{-3} Mpc^{-1})
    rho_plus_P_sigma : Compute shear perturbation (units: eV cm^{-3})
    """

    num_ells_per_bin : int = eqx.field(static=True)

    q_3p = jnp.array([0.913201, 3.37517, 7.79184])
    w_3p = jnp.array([0.0687359, 3.31435, 2.29911])
    q_5p = jnp.array([0.583165, 2.0, 4.0, 7.26582, 13.0])
    w_5p = jnp.array([0.0081201, 0.689407, 2.8063, 2.05156, 0.12681])

    dlfdlq_3p = -q_3p / (1.+jnp.exp(-q_3p)) # Log derivative of fermi-dirac w.r.t. momentum

    name = "MassiveNeutrino"
    is_matter = True

    def __init__(self, first_idx, specs):

        super().__init__(first_idx, specs)
        self.name = "MassiveNeutrino"
        self.is_matter = True
        self.num_ells_per_bin = specs["l_max_massive_nu"] + 1
        self.num_equations = 3 * self.num_ells_per_bin

    def rho(self, lna, args):
        """
        Compute massive neutrino density.

        Parameters:
        -----------
        lna : float or ArrayLike
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float or ArrayLike
            Massive neutrino density (units: eV cm^{-3})
        """
        params = args

        # Ensure lna is at least 1D for broadcasting
        lna_arr = jnp.atleast_1d(lna)  # shape (N,)
        # shape (N,1):
        a = jnp.exp(lna_arr)[:, None]         
        T = params['T_nu_massive'] * params['TCMB0'] / a  
        x = params['m_nu_massive'] / T 

        # q_5p, w_5p are shape (5,), broadcast with (N, 1)
        integrand = (1. + jnp.exp(-self.q_5p)) / self.q_5p**2 \
                    * jnp.sqrt(self.q_5p**2 + x**2)           # (N, 5)

        # Dot product along last axis with w_5p
        integral = jnp.dot(integrand, self.w_5p)               # (N,)

        rho_val = params['N_nu_massive'] * 4. * T[:, 0]**4 / jnp.pi**2 * integral / cnst.hbar**3 / cnst.c**3

        # Remove extra dimension if original input was scalar
        return jnp.squeeze(rho_val) if jnp.ndim(lna) == 0 else rho_val

    def P(self, lna, args):
        """
        Compute massive neutrino pressure.

        Parameters:
        -----------
        lna : float or ArrayLike
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float or ArrayLike
            Massive neutrino pressure (units: eV cm^{-3})
        """
        params = args

        # Ensure lna is at least 1D for broadcasting
        lna_arr = jnp.atleast_1d(lna)          # shape (N,)
        # shape (N,1)
        a = jnp.exp(lna_arr)[:, None] 
        T = params['T_nu_massive'] * params['TCMB0'] / a 
        x = params['m_nu_massive'] / T 

        # q_5p, w_5p are shape (5,), broadcast with (N, 1)
        integrand = (1. + jnp.exp(-self.q_5p)) / jnp.sqrt(self.q_5p**2 + x**2) # (N, 5)

        # Dot product along last axis with w_5p
        integral = jnp.dot(integrand, self.w_5p)               # (N,)

        P_val = params['N_nu_massive'] * 4./3. * T[:, 0]**4 / jnp.pi**2 * integral / cnst.hbar**3 / cnst.c**3

        # Remove extra dimension if original input was scalar
        return jnp.squeeze(P_val) if jnp.ndim(lna) == 0 else P_val

    def y_ini(self, k, tau_ini, args):
        """
        Compute initial conditions for massive neutrino perturbations.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        tau_ini : float
            Initial conformal time (units: Mpc)
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        array
            Initial perturbation mode values (units: 1/Mpc for kPsi1, else dimensionless)
        """
        params = args

        # Initial conditions for massless neutrinos first, needed here.
        R_nu = params['R_nu']
        s2_squared = 1. - 3.*params['K']/k**2

        delta = - (k*tau_ini)**2/3. * (1.-params["om"]*tau_ini/5.) * s2_squared
        theta = - k*(k*tau_ini)**3/36./(4.*R_nu+15.) \
                * (4.*R_nu+11.+12.*s2_squared-3.*(8.*R_nu**2+50.*R_nu+275.)/20./(2.*R_nu+15.)*tau_ini*params["om"]) * s2_squared
        sigma = (k*tau_ini)**2/(45.+12.*R_nu) * (3.*s2_squared-1.) * (1.+(4.*R_nu-5.)/4./(2.*R_nu+15.)*tau_ini*params["om"])

        bins = []
        for i in range(3):
            q = self.q_3p[i]
            # ZZ : Techniclly Psi1 requires epsilon/q = 1/v, but at early times this should be 1. Should check this accuracy!
            first_three = jnp.array([delta/4., theta/3., sigma/2.]) * q / (1.+jnp.exp(-q))
            bins.append(jnp.concatenate((first_three, jnp.zeros(self.num_ells_per_bin - 3))))

        return jnp.concatenate(bins)

    def y_prime(self, k, lna, metric_h_prime, metric_eta_prime, y, args):
        """
        Compute time derivatives of massive neutrino perturbations.

        Parameters:
        -----------
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
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        array
            Time derivatives of perturbation modes (units: 1/Mpc for kPsi1, else dimensionless)
        """
        BG, params, _, _ = args

        a = jnp.exp(lna)
        T = params['T_nu_massive'] * params['TCMB0'] / a
        x = params['m_nu_massive'] / T
        aH  = BG.aH(lna, params)
        tau = BG.tau(lna)

        # Curved-space free-streaming coefficients and l_max truncation,
        # CLASS perturbations.c ncdm hierarchy a la arXiv:1305.3261.
        K = params['K']
        s = curvature_s_l(jnp.arange(self.num_ells_per_bin), k, K)
        cotK_tau = cot_K(tau, K)

        # Iterate through momentum bins
        bins = []
        for i in range(3):
            q = self.q_3p[i]
            epsilon = jnp.sqrt(q**2 + x**2)
            dlnf0_dlnq = -q / (1+jnp.exp(-q))

            # NOTE: The entries are [Psi0, k * Psi1, Psi2, ...]. If accessing Psi1 make sure to divide out k
            L = jnp.arange(self.num_ells_per_bin) + self.first_idx + i*self.num_ells_per_bin
            Psi = y[L]

            Psi0_prime = -q/epsilon/aH*Psi[1] + metric_h_prime/6. * dlnf0_dlnq
            kPsi1_prime = q*k**2/3./epsilon/aH * (Psi[0] - 2.*s[2]*Psi[2])
            Psi2_prime = q*k/5./epsilon/aH * (2.*s[2]*Psi[1]/k - 3.*s[3]*Psi[3]) - s[2]*(metric_h_prime/15. + 2.*metric_eta_prime/5.) * dlnf0_dlnq

            # Intermediate hierarchy, 3<=L<lmax
            lmax = self.num_ells_per_bin - 1
            L_inter = jnp.arange(3, lmax) # Doesn't include lmax.
            Psi_inter_prime = q*k/epsilon/aH/(2*L_inter+1) * (L_inter*s[L_inter]*Psi[L_inter-1] - (L_inter+1)*s[L_inter+1]*Psi[L_inter+1])

            # lmax mode (no s_lmax on the upstream term, as in CLASS)
            Psi_lmax_prime = q*k/aH/epsilon*Psi[lmax-1] - (lmax+1)*cotK_tau/aH*Psi[lmax]

            # Putting it all together
            bins.append(jnp.concatenate((jnp.array([Psi0_prime, kPsi1_prime, Psi2_prime]), Psi_inter_prime, jnp.array([Psi_lmax_prime]))))

        return jnp.concatenate(bins)

    def rho_delta(self, lna, y, args):
        """
        Compute massive neutrino density perturbation.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Density perturbation (units: eV cm^{-3})
        """
        params = args
        a = jnp.exp(lna)
        T = params['T_nu_massive'] * params['TCMB0'] / a  # (N,)
        x = params['m_nu_massive'] / T  # (N,)

        res = 0.
        for i in range(3):
            q = self.q_3p[i]
            w = self.w_3p[i]
            epsilon = jnp.sqrt(q**2 + x**2)
            Psi0 = y[self.first_idx + i*self.num_ells_per_bin]

            res += w*(1.+jnp.exp(-q))*epsilon/q**2 * Psi0
        return params['N_nu_massive'] * res * 4./jnp.pi**2 * T**4 / cnst.hbar**3 / cnst.c**3

    def rho_plus_P_theta(self, lna, y, args):
        """
        Compute massive neutrino velocity perturbation.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Velocity perturbation (units: eV cm^{-3} Mpc^{-1})
        """
        params = args
        a = jnp.exp(lna)
        T = params['T_nu_massive'] * params['TCMB0'] / a  # (N,)
        x = params['m_nu_massive'] / T  # (N,)

        res = 0.
        for i in range(3):
            q = self.q_3p[i]
            w = self.w_3p[i]
            kPsi1 = y[self.first_idx+1 + i*self.num_ells_per_bin]

            res += w*(1.+jnp.exp(-q))/q * kPsi1
        return params['N_nu_massive'] * res * 4./jnp.pi**2 * T**4 / cnst.hbar**3 / cnst.c**3

    def rho_plus_P_sigma(self, lna, y, args):
        """
        Compute massive neutrino shear perturbation.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        y : array
            Perturbation mode values
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Shear perturbation (units: eV cm^{-3})
        """
        params = args
        a = jnp.exp(lna)
        T = params['T_nu_massive'] * params['TCMB0'] / a  # (N,)
        x = params['m_nu_massive'] / T  # (N,)

        res = 0.
        for i in range(3):
            q = self.q_3p[i]
            w = self.w_3p[i]
            epsilon = jnp.sqrt(q**2 + x**2)
            Psi2 = y[self.first_idx+2 + i*self.num_ells_per_bin]

            res += w*(1.+jnp.exp(-q))/epsilon * Psi2
        return params['N_nu_massive'] * res * 8./3./jnp.pi**2 * T**4 / cnst.hbar**3 / cnst.c**3

    def output_perturbations(self, lna, modes, args):
        BG, params = args
        rho  = vmap(self.rho, in_axes=(0, None))(lna, params)   # (Nlna,)
        rhoP = rho + vmap(self.P, in_axes=(0, None))(lna, params)

        rho_delta    = vmap(self.rho_delta,        in_axes=(0, 1, None))(lna, modes, params)  # (Nlna, Nk)
        rho_P_theta  = vmap(self.rho_plus_P_theta, in_axes=(0, 1, None))(lna, modes, params)
        rho_P_sigma  = vmap(self.rho_plus_P_sigma, in_axes=(0, 1, None))(lna, modes, params)

        return {
            "delta": rho_delta   / rho[:, None],
            "theta": rho_P_theta / rhoP[:, None],
            "sigma": rho_P_sigma / rhoP[:, None],
        }


class Baryon(StandardFluid):
    """
    Baryon fluid species implementation.

    Non-relativistic baryons with density and velocity perturbations.

    Required input parameters: params['omega_b'], params['YHe']

    Required derived parameters: params['R_nu'], params['R_b'], params['om']

    Methods:
    --------
    rho : Compute baryon density (units: eV cm^{-3})
    P : Compute baryon pressure (units: eV cm^{-3})
    cs2 : Compute sound speed squared (units: dimensionless)
    mean_mass : Compute mean baryon mass (units: eV)
    y_ini : Compute initial perturbation conditions 
    y_prime : Compute perturbation time derivatives 
    """
    
    name = "Baryon"
    num_equations = 2
    is_matter = True

    def __init__(self, first_idx, specs):
        super().__init__(first_idx, specs)
        self.name = "Baryon"
        self.is_matter = True

    def rho(self, lna, args):
        """
        Compute baryon density.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Baryon density (units: eV cm^{-3})
        """
        params = args
        return params['omega_b'] * (3.*cnst.H0_over_h**2/8./jnp.pi/cnst.G) / jnp.exp(lna)**3

    def P(self, lna, args):
        """
        Compute baryon pressure.

        Parameters:
        ------------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Baryon pressure (units: eV cm^{-3})

        Notes:
        ------
        Baryon pressure is neglected, standard practice for SM baryons.
        """
        return 0.

    def cs2(self, lna, args):
        """
        Compute sound speed squared.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

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
        BG, params, species_list, species_dict = args
        # Get photon class from list
        i = species_dict["Photon"]
        photon = species_list[i]

        Tm = BG.Tm(lna, params) # Baryon temp
        Tg = BG.TCMB(lna, params) # Photon temp
        mu = self.mean_mass(lna, (BG,params))
        R = 4.*photon.rho(lna, params)/3./self.rho(lna, params)

        return Tm/mu * (5./3. - 2./3.*mu*R/cnst.me/BG.aH(lna, params)/BG.tau_c(lna, params) * (Tg/Tm - 1.))

    def mean_mass(self, lna, args):
        """
        Compute mean baryon mass at given redshift.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        float
            Mean baryon mass (units: eV)

        Notes:
        ------
        Defined to be mu = rho_b / n_b = rho_b / (nH + nHe + ne)
        """
        BG, params = args
        denom = (1.+BG.xe(lna))*(1.-params['YHe']) + cnst.mH / cnst.mHe * params['YHe']
        return cnst.mH / denom

    def y_ini(self, k, tau_ini, args):
        """
        Compute initial conditions for baryon perturbations.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        tau_ini : float
            Initial conformal time (units: Mpc)
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        array
            Initial perturbation mode values (units: 1/Mpc for theta, else dimensionless)
        """
        params = args
        s2_squared = 1. - 3.*params['K']/k**2
        delta = -(k*tau_ini)**2/4. * (1.-params["om"]*tau_ini/5.) * s2_squared
        theta = - k**4 * tau_ini**3/36. * (1.-3.*(1.+5.*params['R_b']-params['R_nu'])/20./(1.-params['R_nu'])*params["om"]*tau_ini) * s2_squared
        return jnp.array([delta, theta])

    def y_prime(self, k, lna, metric_h_prime, metric_eta_prime, y, args):
        """
        Compute time derivatives of baryon perturbations.

        Parameters:
        -----------
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
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        array
            Time derivatives of perturbation modes (units: 1/Mpc for theta, else dimensionless)
        """
        BG, params, species_list, species_dict = args
        # Get photon class from list
        i = species_dict["Photon"]
        photon = species_list[i]

        aH = BG.aH(lna, params)
        cs2 = self.cs2(lna, args)
        R = 4.*photon.rho(lna, params)/3./self.rho(lna, params)
        tau_c = BG.tau_c(lna, params)

        delta = y[self.first_idx]
        theta = y[self.first_idx+1]
        theta_g = photon.get_theta(lna, y, args)
        delta_prime = -theta/aH-metric_h_prime/2.
        theta_prime = -theta + cs2*k**2*delta/aH + R/tau_c/aH*(theta_g-theta)

        return jnp.array([delta_prime, theta_prime])

    def output_perturbations(self, lna, modes, args):
        return {
            "delta": modes[self.first_idx],
            "theta": modes[self.first_idx + 1],
        }

class Photon(StandardFluid):
    """
    Photon fluid species implementation.

    Relativistic photons with temperature and polarization Boltzmann hierarchies.

    Required input parameters: params['TCMB0']

    Required derived parameters: params['R_nu'], params['R_nu'], params['om']

    Attributes:
    -----------
    num_F_ell_modes : int
        Number of temperature multipole moments in Boltzmann hierarchy
    num_G_ell_modes : int
        Number of polarization multipole moments in Boltzmann hierarchy

    Methods:
    --------
    rho : Compute photon density (units: eV cm^{-3})
    P : Compute photon pressure (units: eV cm^{-3})
    y_ini : Compute initial perturbation conditions 
    y_prime : Compute perturbation time derivatives 
    """
    num_F_ell_modes : int = eqx.field(static=True)
    num_G_ell_modes : int = eqx.field(static=True)
    name = "Photon"

    def __init__(self, first_idx, specs):
        super().__init__(first_idx, specs)
        self.name = "Photon"
        self.num_F_ell_modes = specs["l_max_g"] + 1
        self.num_G_ell_modes = specs["l_max_pol_g"] + 1
        self.num_equations = self.num_F_ell_modes + self.num_G_ell_modes

    def rho(self, lna, args):
        """
        Compute photon density.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Photon density (units: eV cm^{-3})
        """
        params = args
        a = jnp.exp(lna)
        return jnp.pi**2/15. * params["TCMB0"]**4 / a**4 / (cnst.c * cnst.hbar)**3

    def P(self, lna, args):
        """
        Compute photon pressure.

        Parameters:
        -----------
        lna : float
            Logarithm of scale factor
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        float
            Photon pressure (units: eV cm^{-3})
        """
        params = args
        return self.rho(lna, params)/3.

    def y_ini(self, k, tau_ini, args):
        """
        Compute initial conditions for photon perturbations.

        Parameters:
        -----------
        k : float
            Wavenumber (units: Mpc^{-1})
        tau_ini : float
            Initial conformal time (units: Mpc)
        args : dict
            Cosmological parameters (params)

        Returns:
        --------
        array
            Initial perturbation mode values (units: 1/Mpc for theta, else dimensionless)
        """
        params = args
        s2_squared = 1. - 3.*params['K']/k**2
        delta = - (k*tau_ini)**2/3. * (1.-params["om"]*tau_ini/5.) * s2_squared
        theta = - k**4 * tau_ini**3/36. * (1.-3.*(1.+5.*params['R_b']-params['R_nu'])/20./(1.-params['R_nu'])*params["om"]*tau_ini) * s2_squared
        return jnp.concatenate((jnp.array([delta, theta]), jnp.zeros(self.num_equations - 2)))

    def y_prime(self, k, lna, metric_h_prime, metric_eta_prime, y, args):
        """
        Compute time derivatives of photon perturbations.

        Parameters:
        -----------
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
        args : tuple
            Background cosmology and cosmological parameters (BG, params)

        Returns:
        --------
        array
            Time derivatives of perturbation modes (units: 1/Mpc for theta, else dimensionless)
        """
        BG, params, species_list, species_dict = args
        # Get Baryon from list
        i = species_dict["Baryon"]
        baryon = species_list[i]

        aH    = BG.aH(lna, params)
        tau_c = BG.tau_c(lna, params)
        tau   = BG.tau(lna)

        Flmax = self.num_F_ell_modes-1
        Glmax = self.num_G_ell_modes-1
        F = lax.dynamic_slice(y, (self.first_idx,), (self.num_F_ell_modes,))
        G = lax.dynamic_slice(y, (self.first_idx+self.num_F_ell_modes,), (self.num_G_ell_modes,))
        delta = F[0]
        theta = F[1]
        sigma = F[2]
        theta_b = baryon.get_theta(lna, y, args)

        # Curved-space free-streaming coefficients (all 1 in the flat limit)
        # and generalized cotangent for the l_max truncation; CLASS
        # perturbations.c photon hierarchy a la arXiv:1305.3261.
        K = params['K']
        s = curvature_s_l(jnp.arange(max(self.num_F_ell_modes, self.num_G_ell_modes) + 1), k, K)
        cotK_tau = cot_K(tau, K)

        delta_prime = -4./3./aH*theta - 2./3.*metric_h_prime
        theta_prime = k**2/aH*(delta/4.-s[2]**2*sigma) + (theta_b-theta)/aH/tau_c
        sigma_prime = 4./15./aH*theta - 3./10.*k/aH*s[3]/s[2]*F[3] + 2./15.*metric_h_prime + 4./5.*metric_eta_prime - 9./10./aH/tau_c*sigma + (G[0]+G[2])/20./aH/tau_c/s[2]
        F3_prime    = k/7./aH * (6.*s[3]*s[2]*sigma - 4.*s[4]*F[4]) - F[3]/aH/tau_c

        # Temperature Boltzmann Hierarchy
        L = jnp.arange(4, Flmax) # Excludes the lmax mode
        Fl_prime    = 1./(2.*L+1.)*k/aH * (L*s[L]*F[L-1]-(L+1)*s[L+1]*F[L+1]) - F[L]/aH/tau_c
        Flmax_prime = k/aH*s[Flmax]*F[Flmax-1] - (Flmax+1)*cotK_tau/aH*F[Flmax] - F[Flmax]/aH/tau_c

        # Polarization Boltzmann Hierarchy
        L = jnp.arange(0, Glmax) # Excludes the lmax mode
        Gl_prime    = 1./(2.*L+1.)*k/aH * (L*s[L]*G[L-1]-(L+1)*s[L+1]*G[L+1]) - G[L]/aH/tau_c \
                    + (2.*s[2]*sigma+G[0]+G[2])/2./aH/tau_c * jnp.concatenate((jnp.array([1., 0., 0.2]), jnp.zeros(Glmax-3)))

        Glmax_prime = k/aH*s[Glmax]*G[Glmax-1] - (Glmax+1)*cotK_tau/aH*G[Glmax] - G[Glmax]/aH/tau_c
        return jnp.concatenate((jnp.array([delta_prime, theta_prime, sigma_prime, F3_prime]), Fl_prime, jnp.array([Flmax_prime]), Gl_prime, jnp.array([Glmax_prime])))

    def output_perturbations(self, lna, modes, args):
        return {
            "delta": modes[self.first_idx],
            "theta": modes[self.first_idx + 1],
            "sigma": modes[self.first_idx + 2],
            "G0":    modes[self.first_idx + self.num_F_ell_modes],
            "G2":    modes[self.first_idx + self.num_F_ell_modes + 2],
        }