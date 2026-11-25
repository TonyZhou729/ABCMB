import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .type_helpers import _to_float


def make_shardfn(shard=None, axis=0):
    """
    Make a sharding function that places every array of a pytree on all
    visible GPUs.

    Parameters:
    -----------
    shard: bool or None, optional
        True always attempts to shard, False restricts to one device, None shards if
        multiple devices are automatically detected.  Defaults to None.
    axis: int, optional
         Batch axis of the arrays.  Defaults to 0.

    Returns:
    --------
    tuple (callable, int)
        (shardfn, n_dev). shardfn places every array of a pytree on all visible
        GPUs, split along its batch axis; shardfn is None if not sharding.
        n_dev is the number of GPUs visible to ABCMB.
    """
    try:
        gpus = jax.devices('gpu')
    except Exception:
        gpus = []
    if not (shard or (shard is None and len(gpus) > 1)):
        return None, len(gpus)
    mesh = Mesh(np.asarray(gpus), axis_names=('batch',))
    split = NamedSharding(mesh, PartitionSpec(*([None] * axis), 'batch'))
    replicated = NamedSharding(mesh, PartitionSpec())
    def shardfn(tree):
        return jax.tree_util.tree_map(
            lambda x: jax.device_put(x, split if x.ndim > axis else replicated) if eqx.is_array(x) else x, tree)
    return shardfn, len(gpus)


def stack_batch(params_list, params_dots=None, shard=None):
    """
    Stack per-cosmology parameter dicts, and optionally tangent directions, along
    a leading batch axis for the different stages of computation.

    If sharding, the batch is padded to a multiple of the device count by repeating
    the last cosmology (ABCMB callers drop this padding from their output).

    Parameters:
    -----------
    params_list : list of dict
        Cosmological parameters (including derived parameters), one dict per cosmology
    params_dots : list of dict or None, optional
        One dict per tangent direction: tangents of entries in params_list, each a scalar
        or an array with a leading axis over the number of cosmologies evaluated in 
        parallel.  Missing entries are 0--defaults to None (no directions).
    shard : bool or None, optional
        True always attempts to shard, False restricts to one device, None shards if
        multiple devices are automatically detected.  Defaults to None.

    Returns:
    --------
    tuple
        (params_stack, params_dot, shardfn, dir_shardfn, n_dev):
        params_stack : dict, a single dict with a leading batch axis
        params_dot : dict or None, tangents with a leading direction axis, followed by 
            the batch axis.  None if no directions are provided.
        shardfn, dir_shardfn : callable or None, sharding function for cosmologies 
            (axis 0) and tangents (axis 1), respectively.  None when not sharding.  
            dir_shardfn is None if no directions are provided.
        n_dev : int, number of visible GPUs.
    """
    num_cosmologies = len(params_list)
    shardfn, n_dev = make_shardfn(shard)
    pad = (-num_cosmologies) % n_dev if shardfn is not None else 0
    params_stack = _to_float(jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *(list(params_list) + [params_list[-1]] * pad)))

    params_dot = None
    dir_shardfn = None
    if params_dots is not None: # broadcast, pad, stack
        def tangent(d):
            t = {k: jnp.broadcast_to(jnp.asarray(0. if d.get(k) is None else d[k], v.dtype), (num_cosmologies,)+v.shape[1:])
                for k, v in params_stack.items()}
            return jax.tree_util.tree_map(lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)]), t)
        params_dot = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[tangent(d) for d in params_dots])
    if shardfn is not None:
        params_stack = shardfn(params_stack)
        if params_dot is not None:
            dir_shardfn = make_shardfn(shard, axis=1)[0]
            params_dot = dir_shardfn(params_dot)
    return params_stack, params_dot, shardfn, dir_shardfn, n_dev


def jvp_stage(fn, primals, tangents=None):
    """
    Evaluate fn(*primals) and, given tangent directions, its forward-mode derivative 
    along a stack of directions.

    Parameters:
    -----------
    fn : callable
        Pipeline stage
    primals : tuple of pytrees
        Inputs of fn; their inexact-array (e.g. floats) leaves are differentiated
    tangents : tuple of pytrees or None, optional
        Tangents of the inexact-array leaves of primals, each with a leading direction 
        axis.  Defaults to None: fn is simply evaluated, no tangents are returned.

    Returns:
    --------
    tuple
        (fn(*primals), tangent of its inexact-array leaves with a leading direction 
            axis, or None without tangents)
    """
    if tangents is None:
        return fn(*primals), None
    diffs, statics = zip(*(eqx.partition(p, eqx.is_inexact_array) for p in primals))
    f = lambda *ds: fn(*(eqx.combine(d, s) for d, s in zip(ds, statics)))
    out, out_dot = jax.vmap(lambda *ts: jax.jvp(f, diffs, ts), out_axes=(None, 0))(*tangents)
    return out, eqx.filter(out_dot, eqx.is_inexact_array)
