import functools
import warnings

_REGISTRY = []  # (qualified name, removed_in)

def _parse(v):
    return tuple(int(x) for x in v.split(".")[:3])

def deprecated(since, removed_in, alternative=None):
    def decorate(fn):
        msg = (f"{fn.__qualname__} is deprecated since ABCMB {since} and will be "
                f"removed in {removed_in}.")
        if alternative:
            msg += f" Use {alternative} instead."
        _REGISTRY.append((f"{fn.__module__}.{fn.__qualname__}", removed_in))

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, FutureWarning, stacklevel=2)
            return fn(*args, **kwargs)
        return wrapper
    return decorate