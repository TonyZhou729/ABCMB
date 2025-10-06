import os
import jax

# Make CI runs consistent
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# Enable double precision
jax.config.update("jax_enable_x64", True)

# Uncomment if you want failing fast on NaNs/Infs
jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)
