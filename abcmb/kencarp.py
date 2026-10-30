import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
from typing import Any, ClassVar

jax.config.update("jax_enable_x64", True)

"""
Linearly implicit KenCarp4 solver for the perturbation equations.

The perturbation equations are linear and homogeneous in the state, dy/dlna = A(lna) y, where A
includes the metric constraints and every fluid approximation (UFA, 3EFA, RSA are switches in time,
not in y). An implicit Runge-Kutta stage (I - gamma dt A) x = r is then solved exactly by one linear
solve, with no Newton iterations; A comes from jax.jacfwd of the derivative function.

The tableau is the implicit (ESDIRK) half of Kennedy & Carpenter's ARK4(3)6L[2]SA, as in
diffrax/_solver/kencarp4.py: L-stable, stiffly accurate, with an embedded third-order error estimate.
"""

### KenCarp4 implicit tableau ###
_G = 0.25
_C = np.array([0., 0.5, 83/250, 31/50, 17/20, 1.])
_B = np.array([82889/524892, 0., 15625/83664, 69875/102672, -2260/8211, _G])
_B_HAT = np.array([4586570599/29645900160, 0., 178811875/945068544,
                   814220225/1159782912, -3700637/11593932, 61727/225920])
_B_ERR = _B - _B_HAT
_AI = [
    [],
    [_G],
    [8611/62500, -1743/31250],
    [5012029/34652500, -654441/2922500, 174375/388108],
    [15267082809/155376265600, -71443401/120774400, 730878875/902184768, 2285395/8070912],
    list(_B[:-1]), # Stiffly accurate: the last stage is the solution.
]


def hermite(s, y0, y1, d0, d1):
    """
    Cubic Hermite interpolation on s in [0, 1].

    Parameters:
    -----------
    s : float
        Position within the step, (t - t0)/(t1 - t0)
    y0, y1 : array
        Values at the step ends
    d0, d1 : array
        Derivatives at the step ends, multiplied by the step size

    Returns:
    --------
    array
        Interpolated value
    """
    s2, s3 = s*s, s*s*s
    return (2*s3 - 3*s2 + 1)*y0 + (s3 - 2*s2 + s)*d0 + (-2*s3 + 3*s2)*y1 + (s3 - s2)*d1


class KenCarpInterpolation(diffrax.AbstractLocalInterpolation):
    """
    Dense output within one LinearKenCarp4 step: cubic Hermite through the step ends.

    Needed so that SaveAt(ts=lna) samples the output grid between steps without forcing extra steps.

    Attributes:
    -----------
    t0, t1 : float
        Step start and end
    y0, y1 : array
        State at the step start and end
    f0, f1 : array
        Derivative at the step start and end
    """
    t0 : Any
    t1 : Any
    y0 : Any
    y1 : Any
    f0 : Any
    f1 : Any

    def evaluate(self, t0, t1=None, left=True):
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        dt = self.t1 - self.t0
        s = (t0 - self.t0)/jnp.where(dt == 0, 1., dt)
        return hermite(s, self.y0, self.y1, dt*self.f0, dt*self.f1)


class LinearKenCarp4(diffrax.AbstractSolver):
    """
    Linearly implicit KenCarp4 (ESDIRK) step for linear ODEs dy/dt = A(t) y.

    Each of stages 2-6 builds A(t_i) with jax.jacfwd of the term's vector field and solves
    (I - gamma dt A_i) x_i = y0 + dt sum_j a_ij f_j exactly. The stage derivative is then
    f_i = A_i x_i = (x_i - rhs_i)/(gamma dt), with no extra function evaluation.

    The result is only correct for vector fields linear in y, which all ABCMB perturbation
    equations are. A user species with nonlinear equations needs Kvaerno5.

    Attributes:
    -----------
    filter_error : bool
        Default = True
        Filter the embedded error through (I - gamma dt A)^{-1} (Hairer & Wanner, Solving ODEs II,
        IV.8). The raw embedded difference overstates the error of stiff, decaying modes.

    Methods:
    --------
    step : One step, returning the solution, error estimate and dense output data
    func : Vector field at (t0, y0)
    """
    filter_error : bool = eqx.field(static=True, default=True)

    term_structure : ClassVar = diffrax.AbstractTerm
    interpolation_cls : ClassVar = KenCarpInterpolation

    def order(self, terms):
        return 4

    def error_order(self, terms):
        return 4

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        """
        Advance one step from t0 to t1.

        Parameters:
        -----------
        terms : diffrax.AbstractTerm
            The ODE term, linear in y
        t0, t1 : float
            Step start and end
        y0 : array
            State at t0
        args : tuple
            Passed to the vector field
        solver_state, made_jump : unused

        Returns:
        --------
        tuple
            (y1, error estimate, dense info, solver state, result)
        """
        del solver_state, made_jump
        dt = t1 - t0
        ts = t0 + dt*_C
        gdt = _G*dt
        eye = jnp.eye(y0.shape[0])

        f = [terms.vf(t0, y0, args)] # The first stage is explicit: x_1 = y0.
        for i in range(1, 6):
            # A does not depend on y (linear equations), so differentiating at y0 is exact.
            A = jax.jacfwd(lambda y: terms.vf(ts[i], y, args))(y0)
            rhs = y0 + dt*sum(_AI[i][j]*f[j] for j in range(i))
            x = jnp.linalg.solve(eye - gdt*A, rhs)
            f.append((x - rhs)/gdt)
        y1 = x

        y_error = dt*sum(_B_ERR[j]*f[j] for j in range(6))
        if self.filter_error:
            # jnp.linalg.solve again rather than reusing an LU factorisation: on GPU it is ~2.5x
            # faster than lu_factor + lu_solve for these small batched systems.
            y_error = jnp.linalg.solve(eye - gdt*A, y_error)

        dense_info = dict(y0=y0, y1=y1, f0=f[0], f1=f[5])
        return y1, y_error, dense_info, None, diffrax.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


class WindowedDtmax(diffrax.AbstractAdaptiveStepSizeController):
    """
    Wraps an adaptive step size controller and caps the step inside time windows.

    A step starting at t in [lo, hi) is at most dtmax. This lets the global cap stay loose while
    recombination is resolved finely.

    Attributes:
    -----------
    controller : diffrax.AbstractAdaptiveStepSizeController
        The wrapped controller, e.g. diffrax.PIDController
    windows : tuple
        Tuple of (lo, hi, dtmax)
    """
    controller : Any
    windows : tuple = eqx.field(static=True)

    @property
    def rtol(self):
        return self.controller.rtol

    @property
    def atol(self):
        return self.controller.atol

    @property
    def norm(self):
        return self.controller.norm

    def _cap(self, t0, t1):
        for lo, hi, dtmax in self.windows:
            t1 = jnp.where((t0 >= lo) & (t0 < hi), jnp.minimum(t1, t0 + dtmax), t1)
        return t1

    def wrap(self, direction):
        return eqx.tree_at(lambda c: c.controller, self, self.controller.wrap(direction))

    def init(self, terms, t0, t1, y0, dt0, args, func, error_order):
        t1_next, state = self.controller.init(terms, t0, t1, y0, dt0, args, func, error_order)
        return self._cap(t0, t1_next), state

    def adapt_step_size(self, t0, t1, y0, y1_candidate, args, y_error, error_order, controller_state):
        keep, t0_next, t1_next, made_jump, state, result = self.controller.adapt_step_size(
            t0, t1, y0, y1_candidate, args, y_error, error_order, controller_state)
        return keep, t0_next, self._cap(t0_next, t1_next), made_jump, state, result
