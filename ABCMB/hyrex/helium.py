import jax.numpy as jnp
from jax import jit, config, lax, grad
import equinox as eqx

from diffrax import diffeqsolve, SaveAt, ODETerm, Kvaerno3, PIDController, ForwardMode, Event

from ABCMB import constants as cnst
from . import recomb_functions
from .array_with_padding import array_with_padding
config.update("jax_enable_x64", True)


class helium_model(eqx.Module):
    """
    Helium recombination model implementation.

    Computes helium ionization fraction evolution through multiple phases:
    HeII+III equilibrium, post-Saha HeII expansion, and full HeII recombination.

    Methods:
    --------
    get_helium_history : Compute full helium recombination history (units: dimensionless)
    xesaha_HeII_III : Compute HeII+III equilibrium phase (units: dimensionless)
    post_saha_xHeII : Compute post-Saha HeII expansion (units: dimensionless)
    solve_HeII_full : Solve full HeII recombination (units: dimensionless)
    xHeII_post_Saha : Compute HeII fraction in post-Saha regime (units: dimensionless)
    xH1_Saha : Compute neutral hydrogen fraction in Saha equilibrium (units: dimensionless)
    helium_dxHeIIdlna : Compute HeII recombination rate (units: dimensionless)
    xe_derivative_HeII : Compute HeII derivative for ODE integration (units: dimensionless)
    """

    integration_spacing : jnp.float64
    lna_axis_4Heequil : jnp.array

    concrete_axis_size : jnp.array
    concrete_axis_size_postSahaHe : jnp.array

    def __init__(self,lna_axis_4Heequil,integration_spacing = 5.0e-4, Nsteps=800, Nsteps_postSahaHe=4000,z0=8000., z1=20.):
        """
        Initialize helium recombination model.

        Parameters:
        -----------
        lna_axis_4Heequil : array
            Log scale factor grid for HeII+III equilibrium phase
        integration_spacing : float, optional
            Step size for integration (default: 5.0e-4)
        Nsteps : int, optional
            Maximum number of integration steps (default: 800)
        Nsteps_postSahaHe : int, optional
            Maximum steps for post-Saha HeII phase (default: 4000)
        z0 : float, optional
            Initial redshift (default: 8000.)
        z1 : float, optional
            Final redshift (default: 20.)
        """
        self.integration_spacing = integration_spacing
        self.concrete_axis_size_postSahaHe = jnp.zeros(Nsteps_postSahaHe)

        # Define time axes
        self.lna_axis_4Heequil = lna_axis_4Heequil
        self.concrete_axis_size = jnp.zeros(Nsteps)

    def __call__(self, BG, rtol=1e-6, atol=1e-9,solver=Kvaerno3(),max_steps=1024):
        """
        Compute helium recombination history.

        Parameters:
        -----------
        BG : cosmology.Background
            Background cosmology module
        rtol : float, optional
            Relative tolerance for ODE solver (default: 1e-6)
        atol : float, optional
            Absolute tolerance for ODE solver (default: 1e-9)
        solver : diffrax.Solver, optional
            ODE solver instance (default: Kvaerno3())
        max_steps : int, optional
            Maximum solver steps (default: 1024)

        Returns:
        --------
        tuple
            (xe_4He, lna_4He) - helium ionization fraction and log scale factor
        """
        return self.get_helium_history(BG, rtol, atol, solver, max_steps)
    
    def get_helium_history(self, BG, rtol=1e-6, atol=1e-9,solver=Kvaerno3(),max_steps=1024):
        """
        Compute complete helium recombination history through all phases.

        Sequentially computes helium ionization fraction through HeII+III equilibrium,
        post-Saha HeII expansion, and full HeII recombination phases.

        Parameters:
        -----------
        BG : cosmology.Background
            Background cosmology module
        rtol : float, optional
            Relative tolerance for ODE solver (default: 1e-6)
        atol : float, optional
            Absolute tolerance for ODE solver (default: 1e-9)
        solver : diffrax.Solver, optional
            ODE solver instance (default: Kvaerno3())
        max_steps : int, optional
            Maximum solver steps (default: 1024)

        Returns:
        --------
        tuple
            (xe_4He, lna_4He) containing helium ionization fraction evolution
            and log scale factor grid
        """
        # Calculate omega_rad today using input Neff.
        # omega_rad = cosmology.omega_rad0(Neff)

        # Compute xe at different phases

        # Give it a large enough array of lna to work with
        xe_output_4He_equil, lna_output_4He_equil = self.xesaha_HeII_III(self.lna_axis_4Heequil, BG)
        
        xe_output_4He_equil_obj = array_with_padding(xe_output_4He_equil)
        lna_output_4He_equil_obj = array_with_padding(lna_output_4He_equil)

        # this one MUST start shifted by one redshift bin to avoid overlapping redshifts
        xe_output_4He_postSaha, lna_output_4He_postSaha = self.post_saha_xHeII(lna_output_4He_equil_obj.lastval + self.integration_spacing, BG)

        xe_4He_equil_post = xe_output_4He_equil_obj.concat(array_with_padding(xe_output_4He_postSaha))
        lna_4He_equil_post = lna_output_4He_equil_obj.concat(array_with_padding(lna_output_4He_postSaha))

        xe_output_4He_full, lna_output_4He_full = self.solve_HeII_full(
            lna_4He_equil_post.lastval, xe_4He_equil_post.lastval, BG, rtol=1e-6, atol=1e-9,solver=Kvaerno3(),max_steps=4096)

        xe_4He = xe_4He_equil_post.concat(array_with_padding(xe_output_4He_full))
        lna_4He = lna_4He_equil_post.concat(array_with_padding(lna_output_4He_full))

        return (xe_4He, lna_4He)



    ######################  HELIUM RECOMBINATION  ######################

    # HYREC-2's helium.c expects T in K, but we use eV instead, hence proliferation of
    # kB's.

    # High tempateratures (z >~ 4000).  Function to calculate xe in He II + III equilibrium
    # We use this form until xHeIII is 1e-9
    def xesaha_HeII_III(self, lna_axis, BG, threshold=1e-9):
        """
        Compute xe in HeII+III equilibrium phase.

        Calculates ionization fraction assuming equilibrium between HeII and HeIII
        until HeIII fraction drops below threshold.

        Parameters:
        -----------
        lna_axis : array
            Log scale factor grid
        BG : cosmology.Background
            Background cosmology module
        threshold : float, optional
            Threshold for HeIII fraction to stop calculation (default: 1e-9)

        Returns:
        --------
        tuple
            (xe_output, lna_output) - ionization fraction and log scale factor arrays
        """
        # Pre-allocate xe_output
        xe_output = jnp.ones_like(lna_axis)*jnp.inf
        lna_output = jnp.ones_like(lna_axis)*jnp.inf
        iz = int(0)
        xe = 1. # arbitrary IC
        stop = False

        def compute_xe(carry):
            xe_output, lna_output, xe, iz, stop = carry

            lna = lna_axis[iz]

            # Cosmological parameters
            TCMB = BG.TCMB(lna)
            nH = BG.nH(lna)
 
            # compute xHeIII
            fHe = BG.params['YHe']/(1.-BG.params['YHe'])/3.97153 # abundance of helium by number
            # Saha ratio xe * xHeIII / xHeII
            s = 2.414194e15 * TCMB/cnst.kB * jnp.sqrt(TCMB/cnst.kB) * jnp.exp(-631462.7 / (TCMB/cnst.kB)) / nH
            xHeIII = 2 * s * fHe / (1 + s + fHe) / (1 + jnp.sqrt(1 + 4 * s * fHe / (1 + s + fHe)**2))
            xe = 1 + fHe + xHeIII

            # Store current xe value in the output array
            xe_output = xe_output.at[iz].set(xe)
            lna_output = lna_output.at[iz].set(lna)

            # Check difference
            stop = xHeIII < threshold  # Stop when xHeIII > threshold

            # Increment index
            iz = iz + 1

            return (xe_output, lna_output, xe, iz, stop)

        def stop_condition(state):
            _, _, _, iz, stop = state
            return (iz < lna_axis.size) & (~stop)  # Continue until stop condition is met or we run out of space

        # Initial state: (xe_output, xe, iz, stop flag)
        initial_state = (xe_output, lna_output, xe, iz, stop)

        # Run the while loop until the stop condition is met
        final_state = lax.while_loop(stop_condition, compute_xe, initial_state)

        # Unpack the final state
        xe_output_final, lna_output_final, _, _, _ = final_state

        # Return the electron fraction array and lna
        return xe_output_final, lna_output_final


    def xHeII_post_Saha(self, lna, BG):
        """
        Compute HeII fraction in post-Saha regime.

        Calculates HeII fraction using Saha equilibrium between HeI and HeII.

        Parameters:
        -----------
        lna : float
            Log scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            HeII fraction (units: dimensionless)
        """
        fHe = BG.params['YHe']/(1.-BG.params['YHe'])/3.97153

        TCMB = BG.TCMB(lna)
        nH = BG.nH(lna)

        # Saha ratio xe * xHeII / xHeI
        s = 4 * 2.414194e15 * TCMB/cnst.kB * jnp.sqrt(TCMB/cnst.kB) * jnp.exp(-285325. / (TCMB/cnst.kB)) / nH
        xHeII = 2 * s * fHe / (1 + s) / (1 + jnp.sqrt(1 + 4 * s * fHe / (1 + s)**2))

        return xHeII

    def xH1_Saha(self, lna, BG):
        """
        Compute neutral hydrogen fraction in Saha equilibrium.

        Calculates neutral hydrogen fraction assuming Saha equilibrium
        between hydrogen ionization and recombination.

        Parameters:
        -----------
        lna : float
            Log scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Neutral hydrogen fraction (units: dimensionless)
        """
        TCMB = BG.TCMB(lna)
        nH = BG.nH(lna)    
        xHeII = self.xHeII_post_Saha(lna, BG)
        s = 2.4127161187130e15* TCMB/cnst.kB * jnp.sqrt(TCMB/cnst.kB)*jnp.exp(-157801.37882/(TCMB/cnst.kB))/nH
        xH1 = jnp.where(s>1e5,(1.+xHeII)/s - (xHeII**2 + 3.*xHeII + 2.)/s**2,\
            jnp.where(s==0,1,1.-2./(1.+ xHeII/s + jnp.sqrt((1.+ xHeII/s)*(1.+ xHeII/s) +4./s))))
        return xH1

    # xHeII near equilibrium
    # Returns xHeII-->xe (98 of 1011.3758).  Integrate until delta x_e ~ 1e-5
    def post_saha_xHeII(self, starting_lna, BG, threshold=1e-5):
        """
        Compute post-Saha HeII expansion phase.

        Calculates ionization fraction including corrections to Saha equilibrium
        until deviations exceed threshold.

        Parameters:
        -----------
        starting_lna : float
            Initial log scale factor
        BG : cosmology.Background
            Background cosmology module
        threshold : float, optional
            Threshold for deviation from Saha (default: 1e-5)

        Returns:
        --------
        tuple
            (xe_output, lna_output) - ionization fraction and log scale factor arrays
        """
        # Pre-allocate xe_output
        xe_output = jnp.ones_like(self.concrete_axis_size_postSahaHe)*jnp.inf
        lna_output = jnp.ones_like(self.concrete_axis_size_postSahaHe)*jnp.inf
        iz = int(0)
        xe = 1. # arbitrary
        stop = False

        def compute_xe(carry):
            xe_output, lna_output, xe, iz, stop = carry

            lna = starting_lna + iz*self.integration_spacing
            
            xHeII = self.xHeII_post_Saha(lna, BG)
            xH1 = self.xH1_Saha(lna, BG)
            xe_saha = 1 - xH1 + xHeII
            
            # Do post saha expansion.  Assume all hydrogen is ionized.
            grad_dxedlna_func = grad(self.helium_dxHeIIdlna, argnums=0) 
            grad_dxedlna = grad_dxedlna_func(xe_saha, lna, BG)
            dxe_Saha_dlna = grad(self.xHeII_post_Saha,argnums=0)(lna, BG)
            xe = xe_saha + dxe_Saha_dlna / grad_dxedlna

            # Store current xe value in the output array
            xe_output = xe_output.at[iz].set(xe)
            lna_output = lna_output.at[iz].set(lna)

            # Check difference
            diff = jnp.abs(xe_saha - xe)
            stop = diff > threshold  # Stop when diff < threshold

            # Increment index
            iz = iz + 1

            return (xe_output, lna_output, xe, iz, stop)

        def stop_condition(state):
            _, _, _, iz, stop = state
            return (iz < self.concrete_axis_size_postSahaHe.size) & (~stop)  # Continue until stop condition is met or we run out of space

        # Initial state: (xe_output, xe, iz, stop flag)
        initial_state = (xe_output, lna_output, xe, iz, stop)

        # Run the while loop until the stop condition is met
        final_state = lax.while_loop(stop_condition, compute_xe, initial_state)

        # Unpack the final state
        xe_output_final, lna_output_final, _, _, _ = final_state

        # Return the electron fraction array and the stopping `lna` value
        return xe_output_final, lna_output_final

    def helium_dxHeIIdlna(self, xe, lna, BG):
        """
        Compute HeII recombination rate.

        Calculates rate of change of HeII ionization fraction including
        detailed atomic physics and escape probabilities.

        Parameters:
        -----------
        xe : float
            Current total ionization fraction
        lna : float
            Log scale factor
        BG : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            HeII recombination rate dxHeII/dlna (units: dimensionless)
        """

        fHe = BG.params['YHe']/(1.-BG.params['YHe'])/3.97153 # abundance of helium by number

        # cosmology
        #lna = -jnp.log(1+z)
        TCMB = BG.TCMB(lna)      # eV
        nH = BG.nH(lna)  # hydrogen number density, 1/cm^3
        H = BG.H(lna)  # Hubble parameter, 1/s
        GammaC = recomb_functions.Gamma_compton(xe, TCMB, BG.params['YHe'])  # Compton scattering rate, 1/s

        # compute xH1 in Saha equilibrium, xHeII in post-saha
        xH1 = self.xH1_Saha(lna, BG)
        # use xe  = xHeII + (1.-xH1)
        xHeII = xe - (1.-xH1)

        # Saha ratio and abundances
        s0 = 2.414194e15 * TCMB/cnst.kB * jnp.sqrt(TCMB/cnst.kB) / nH * 4
        s = s0 * jnp.exp(-285325. / (TCMB/cnst.kB))
        y2s = jnp.exp(46090. / (TCMB/cnst.kB)) / s0
        y2p = jnp.exp(39101. / (TCMB/cnst.kB)) / s0 * 3
        
        # Continuum opacity and optical depth
        etacinv =  9.15776e22 * H / (nH* xH1)
        g2pinc = (
            1.976e6 / (1 - jnp.exp(-6989. / (TCMB/cnst.kB))) +
            6.03e6 / (jnp.exp(19754. / (TCMB/cnst.kB)) - 1) +
            1.06e8 / (jnp.exp(21539. / (TCMB/cnst.kB)) - 1) +
            2.18e6 / (jnp.exp(28496. / (TCMB/cnst.kB)) - 1) +
            3.37e7 / (jnp.exp(29224. / (TCMB/cnst.kB)) - 1) +
            1.04e6 / (jnp.exp(32414. / (TCMB/cnst.kB)) - 1) +
            1.51e7 / (jnp.exp(32781. / (TCMB/cnst.kB)) - 1)
        )
        
        # Optical depth and escape probability
        tau2p = jnp.float64(4.277e-8 * nH / H * (fHe - xHeII))
        dnuline = g2pinc * tau2p / (4 * jnp.pi**2)
        tauc = dnuline / etacinv
        enh = jnp.sqrt(1 + jnp.pi**2 * tauc) + 7.74 * tauc / (1 + 70 * tauc)
        pesc = enh / tau2p
        
        # Total decay rate
        ydown = (50.94 * y2s) + (1.7989e9 * y2p * pesc)
        
        # Recombination rate
        return ydown * ((fHe - xHeII) * s - xHeII * (xHeII + 1 - xH1)) / H
    

    def xe_derivative_HeII(self, lna, state, args):
        """
        Compute HeII derivative for ODE integration.

        Derivative function for HeII ionization fraction evolution
        used in ODE integration with diffrax.

        Parameters:
        -----------
        lna : float
            Log scale factor
        state : float
            Current HeII ionization state
        args : cosmology.Background
            Background cosmology module

        Returns:
        --------
        float
            Time derivative of HeII fraction (units: dimensionless)
        """
        
        BG = args
        #z = 1. / jnp.exp(lna) - 1.
        # use xe  = xHeII + (1.-xH1)
        xe = state + self.xH1_Saha(lna, BG)

        return self.helium_dxHeIIdlna(xe, lna, BG)

    def solve_HeII_full(self, starting_lna, xe0, BG, rtol=1e-6, atol=1e-9,solver=Kvaerno3(),max_steps=1024):
        """
        Solve full HeII recombination evolution.

        Integrates HeII recombination including detailed atomic physics
        until HeII fraction becomes negligible.

        Parameters:
        -----------
        starting_lna : float
            Initial log scale factor
        xe0 : float
            Initial ionization fraction
        BG : cosmology.Background
            Background cosmology module
        rtol : float, optional
            Relative tolerance (default: 1e-6)
        atol : float, optional
            Absolute tolerance (default: 1e-9)
        solver : diffrax.Solver, optional
            ODE solver (default: Kvaerno3())
        max_steps : int, optional
            Maximum steps (default: 1024)

        Returns:
        --------
        tuple
            (xe_output, lna_output) - ionization fraction and log scale factor arrays
        """

        # Initial conditions
        TCMB_init = BG.TCMB(starting_lna)  # Initial matter temperature
        initial_state = jnp.array([xe0])
        term = ODETerm(self.xe_derivative_HeII)

        t0 = starting_lna
        t1 = jnp.inf

        # don't want to double count the boundary lna, so start saving after one step
        t_arr = jnp.linspace(t0+self.integration_spacing, t0+max_steps*self.integration_spacing, max_steps)

        save_at = SaveAt(ts=t_arr)
        adjoint=ForwardMode()

        def He_check(t, y, args, **kwargs):
            lna = t
            xH1 = self.xH1_Saha(lna, BG)

            # use xe  = xHeII + (1.-xH1)
            # y is current state, [0] flattens jnp.array
            xHeII = y[0] - (1.-xH1)
            return xHeII < 1e-4
        
        event = Event(He_check)

        sol = diffeqsolve(
            term, solver, t0=t0, t1=t1, dt0=1e-3, 
            y0=initial_state, 
            args=BG,
            stepsize_controller=PIDController(rtol, atol),saveat=save_at,
            event = event,
            adjoint=adjoint
        )
        
        xe_output = sol.ys[:, 0]  
        lna_output = sol.ts

        return xe_output, lna_output


