import warnings

from .utils.constants import *
from .utils.deprecation import _REGISTRY

warnings.warn("abcmb.constants is deprecated since ABCMB 1.0.0 and will be removed in 1.2.0. "
              "Use abcmb.utils.constants instead.", FutureWarning, stacklevel=2)
_REGISTRY.append(("abcmb.constants", "1.2.0"))
