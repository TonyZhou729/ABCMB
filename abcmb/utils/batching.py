import numpy as np
import jax
import equinox as eqx
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def make_shardfn(shard=None):
    """
    Make a sharding function that places every array of a pytree on all
    visible GPUs.

    Parameters:
    -----------
    shard: bool or None, optional
        True always attempts to shard, False restricts to one device, None shards if
        multiple devices are automatically detected.  Defaults to None.

    Returns:
    --------
    tuple (callable, int)
        (shardfn, n_dev). shardfn places every array of a pytree on all visible
        GPUs, split along its leading (batch) axis; shardfn is None if not sharding.
        n_dev is the number of GPUs visible to ABCMB.
    """
    try:
        gpus = jax.devices('gpu')
    except Exception:
        gpus = []
    if not (shard or (shard is None and len(gpus) > 1)):
        return None, len(gpus)
    mesh = Mesh(np.asarray(gpus), axis_names=('batch',))
    split, replicated = NamedSharding(mesh, PartitionSpec('batch')), NamedSharding(mesh, PartitionSpec())

    def shardfn(tree):
        return jax.tree_util.tree_map(
            lambda x: jax.device_put(x, split if x.ndim >= 1 else replicated) if eqx.is_array(x) else x, tree)
    return shardfn, len(gpus)


def jvp_stage(fn, primals, tangents):
    """
    Evaluate fn(*primals) and its forward-mode derivative along a stack of directions.

    Parameters:
    -----------
    fn : callable
        Pipeline stage
    primals : tuple of pytrees
        Inputs of fn; their inexact-array leaves are differentiated
    tangents : tuple of pytrees
        Tangents of the inexact-array leaves of primals, each with a leading direction axis

    Returns:
    --------
    tuple
        (fn(*primals), tangent of its inexact-array leaves with a leading direction axis)
    """
    diffs, statics = zip(*(eqx.partition(p, eqx.is_inexact_array) for p in primals))
    f = lambda *ds: fn(*(eqx.combine(d, s) for d, s in zip(ds, statics)))
    out, out_dot = jax.vmap(lambda *ts: jax.jvp(f, diffs, ts), out_axes=(None, 0))(*tangents)
    return out, eqx.filter(out_dot, eqx.is_inexact_array)
