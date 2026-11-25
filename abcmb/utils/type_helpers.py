import jax
import jax.numpy as jnp

def _to_float(tree):
    """
    Cast int/bool leaves to float64.

    Parameters:
    -----------
    tree: pytree
        Pytree whose leaves are to be converted to float.
    
    Returns:
    --------
    array
        Array of leaves as floats
        
    """
    def cast(v):
        arr = jnp.asarray(v)
        return arr.astype(jnp.float64) if arr.dtype.kind in 'iub' else arr

    return jax.tree_util.tree_map(cast, tree)

    