from .utils import interpolators, quadrature
from .utils.deprecation import deprecated


@deprecated(since="1.0.0", removed_in="1.2.0", alternative="abcmb.utils.interpolators.fast_interp")
def fast_interp(x, xp_min, xp_max, fp):
    return interpolators.fast_interp(x, xp_min, xp_max, fp)


@deprecated(since="1.0.0", removed_in="1.2.0", alternative="abcmb.utils.interpolators.fast_interp_cubic")
def fast_interp_cubic(x, xp_min, xp_max, fp):
    return interpolators.fast_interp_cubic(x, xp_min, xp_max, fp)


@deprecated(since="1.0.0", removed_in="1.2.0", alternative="abcmb.utils.interpolators.loglog_interp")
def loglog_interp(x, xp, fp):
    return interpolators.loglog_interp(x, xp, fp)


@deprecated(since="1.0.0", removed_in="1.2.0", alternative="abcmb.utils.interpolators.bilinear_interp")
def bilinear_interp(x, y, z, xq, yq):
    return interpolators.bilinear_interp(x, y, z, xq, yq)


@deprecated(since="1.0.0", removed_in="1.2.0", alternative="abcmb.utils.quadrature.gauss_legendre_weights")
def gauss_legendre_weights(n, tol=1.e-16, max_it=50):
    return quadrature.gauss_legendre_weights(n, tol=tol, max_it=max_it)
