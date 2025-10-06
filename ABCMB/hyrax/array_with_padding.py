import jax.numpy as jnp
from jax import jit, config, lax, grad
from jax.experimental import host_callback as hcb
from jax import debug
import equinox as eqx


class array_with_padding(eqx.Module):
    """
    Array container with automatic padding management.

    Manages arrays with infinite padding, tracking valid data boundaries
    and providing concatenation operations for sequential data assembly.

    Methods:
    --------
    concat : Concatenate with another array_with_padding instance (units: same as input)
    """

    arr : jnp.array
    padding_size : int
    lastnum : int
    lastval : jnp.float64

    def __init__(self,arr):
        """
        Initialize array with padding management.

        Automatically detects padding boundaries and computes metadata
        for efficient array operations.

        Parameters:
        -----------
        arr : array
            Input array with infinite values used as padding
        """
        self.arr = arr

        self.lastnum = jnp.argmax(jnp.isinf(arr)*1)-1
        self.lastval = arr[self.lastnum]
        self.padding_size = arr.size-jnp.argmax(jnp.isinf(arr)*1)

    def __call__(self):
        """
        Return the underlying array.

        Returns:
        --------
        array
            The stored array with padding
        """
        return self.arr

    def concat(self,other_arr):
        """
        Concatenate with another array_with_padding instance.

        Combines arrays while properly managing padding boundaries.
        The current array appears first in the concatenation.

        Parameters:
        -----------
        other_arr : array_with_padding
            Another array_with_padding instance to concatenate

        Returns:
        --------
        array_with_padding
            New instance containing concatenated arrays with updated padding

        Raises:
        -------
        TypeError
            If other_arr is not an array_with_padding instance
        """

        if not isinstance(other_arr, array_with_padding):
            raise TypeError("Can only concatenate with another array_with_padding instance.")
        
        x = self.arr
        y = other_arr.arr
        padding_size = self.padding_size
        z = jnp.ones(x.size + y.size)*jnp.inf # neither of these is a tracer!!!
        z = z.at[0:x.size].set(x)
        concatenated_arr = lax.dynamic_update_slice(z,y,[x.size-padding_size])
        return array_with_padding(concatenated_arr)