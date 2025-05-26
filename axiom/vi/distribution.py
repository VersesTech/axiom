import inspect
from typing import Optional, Union, Tuple

import jax.numpy as jnp
import jax.tree_util as jtu
from jax.tree_util import register_pytree_node
from jaxtyping import Array

from numpy.core.numeric import normalize_axis_tuple

from axiom.vi import ArrayDict, utils


class Distribution:
    """Base class for probability distributions"""

    dim: int
    event_dim: int
    batch_dim: int
    default_event_dim: int
    event_shape: tuple[int]
    batch_shape: tuple[int]
    pytree_data_fields = ()
    pytree_aux_fields = (
        "dim",
        "default_event_dim",
        "batch_dim",
        "event_dim",
        "batch_shape",
        "event_shape",
    )

    def __init__(self, default_event_dim: int, batch_shape: tuple, event_shape: tuple):
        self.event_dim = len(event_shape)
        self.batch_dim = len(batch_shape)
        self.event_shape = event_shape
        self.batch_shape = batch_shape
        self.default_event_dim = default_event_dim
        self.dim = self.event_shape[-default_event_dim] if len(self.event_shape) > 0 else 0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    @property
    def shape(self):
        return self.batch_shape + self.event_shape

    def to_event(self, n: int) -> jnp.ndarray:
        """
        Converts the distribution to an event distribution.
        """
        if n > 0:
            event_shape = self.batch_shape[-n:] + self.event_shape
            batch_shape = self.batch_shape[:-n]
        else:
            event_shape = self.batch_shape + self.event_shape
            batch_shape = ()

        return self.__class__(self.default_event_dim, batch_shape, event_shape)

    def get_sample_dims(self, data: jnp.ndarray) -> list[int]:
        """
        Returns the sample dimensions of the data.
        """
        sample_shape = self.get_sample_shape(data)
        return list(range(len(sample_shape)))

    def get_sample_shape(self, data: jnp.ndarray) -> tuple[int]:
        """
        Returns the sample shape of the data.
        """
        return data.shape[: -self.event_dim - self.batch_dim]

    def get_batch_shape(self, data: jnp.ndarray) -> tuple[int]:
        """
        Returns the batch shape of the data.
        """
        return data.shape[-self.event_dim - self.batch_dim : -self.event_dim]

    def get_event_dims(self) -> list[int]:
        """
        Return the event dimensions of the array.
        """
        return list(range(-self.event_dim, 0))

    def sum_events(self, x: jnp.ndarray, keepdims: bool = False) -> jnp.ndarray:
        """
        Sums over the event dimensions of the array.
        """
        return x.sum(range(-self.event_dim, 0), keepdims=keepdims)

    def sum_default_events(self, x: jnp.ndarray, keepdims: bool = False) -> jnp.ndarray:
        """
        Sums over the default event dimensions of the array.
        """
        return x.sum(range(-self.default_event_dim, 0), keepdims=keepdims)

    def expand_event_dims(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Adds event dimensions to the array.
        """
        return x.reshape(x.shape + (1,) * self.event_dim)

    def expand_default_event_dims(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Adds event dimensions to the array.
        """
        return x.reshape(x.shape + (1,) * self.default_event_dim)

    def expand_batch_dims(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Adds batch dimensions to the array.
        """
        return x.reshape(x.shape[: -self.event_dim] + (1,) * self.batch_dim + x.shape[-self.event_dim :])

    def expand_batch_shape(self, batch_relative_axes: Union[int, Tuple[int]]):
        """
        Returns a new Distribution object with the expanded batch shape. Expands all Arrays into a larger batch shape, by inserting singleton dimensions
        at the specified axes. Similar to existing `expand_batch_dims` except it operates on all the leaves of the Distribution object, rather than just
        an input tensor. This includes nested sub-distributions within `self`, since the function is recursively called on any leaves of `self` that themselves are `Distribution` instances

        The `batch_relative_axes` argument is assumed to be relative to `self.event_dim`. For example, if `axes == -1`, then you would expand the last batch dimension with a singleton dimension.
        """

        batch_relative_axes = (batch_relative_axes,) if isinstance(batch_relative_axes, int) else batch_relative_axes

        # add the extra batch dimensions, if A) the leaf is an Array and B) if the Array you're trying to reshape has shape-compatible dimensions
        def expand_if_possible(x, batch_dim, event_dim, batch_ax, abs_ax):
            if isinstance(x, Array):
                if x.ndim == batch_dim + event_dim:
                    return jnp.expand_dims(x, abs_ax)
                if x.ndim == batch_dim:
                    return jnp.expand_dims(x, batch_ax)
            return x

        # convert axes into absolute indices (that account for event_dim), if they are negative (i.e., -1 becomes -3)
        absolute_axes = tuple([ax - self.event_dim if ax < 0 else ax for ax in batch_relative_axes])

        # An alternative version which might be faster, but requires implementing custom `get_parameters()` and `get_distributions()` methods for each Distribution class
        # Another idea: eplace these two lines with lines which retrieve tree leaves but filter out A) parameters (which include Arrays and ArrayDicts -- excluding Distributions) and B) Distributions
        # params = self.get_parameters()
        # sub_dists = self.get_distributions()

        ## version that is more generic but potentially much slower, since it loops over all the leaves in pytree_data_fields
        data_fields, aux_fields = self.tree_flatten()
        expanded_data_fields = []
        for leaf in data_fields:
            if isinstance(leaf, Distribution):
                exp_leaf = leaf.expand_batch_shape(batch_relative_axes)
            elif isinstance(leaf, ArrayDict):
                exp_leaf = jtu.tree_map(
                    lambda x: expand_if_possible(x, self.batch_dim, self.event_dim, batch_relative_axes, absolute_axes),
                    leaf,
                )
            else:
                exp_leaf = expand_if_possible(leaf, self.batch_dim, self.event_dim, batch_relative_axes, absolute_axes)

            expanded_data_fields.append(exp_leaf)

        # Update batch_shape and batch_dim to account for the new batch dimension
        new_batch_dim = len(batch_relative_axes) + self.batch_dim
        adjusted_axes = normalize_axis_tuple(batch_relative_axes, new_batch_dim)

        # We have to add singleton dimensions in the right places in the new_batch_shape
        shape_it = iter(self.batch_shape)
        new_batch_shape = [1 if ax in adjusted_axes else next(shape_it) for ax in range(new_batch_dim)]

        replace = {"batch_shape": tuple(new_batch_shape)}
        unsqueezed_dist = self.tree_unflatten_and_replace(aux_fields, expanded_data_fields, replace)

        return unsqueezed_dist

    def swap_axes(self, axis1: int, axis2: int):
        """
        Swaps the axes of the component distributions of a Pytree.
        NOTE: The axis1 and axis2 arguments should be made relative to batch shape (see Issue #109).
        """

        # add the extra batch dimensions, if A) the leaf is an Array and B) if the Array you're trying to reshape has shape-compatible dimensions
        def swap_axes_of_leaf(x, ax1, ax2):
            if isinstance(x, Array):
                if (
                    x.ndim >= self.batch_dim + self.event_dim
                ):  # the greater or equal to accounts for cases when certain leaves have `sample_shape` (e.g. distributions that represent samples)
                    return jnp.swapaxes(x, ax1, ax2)
                elif (
                    x.ndim > 1 and x.ndim >= self.batch_dim
                ):  # this is the case of elements that lack event-dimensions, e.g. residuals
                    # if the axes are negative (i.e. relative to the end of the array), we need to correct them by the number of event dimensions
                    ax1 = ax1 + self.event_dim if ax1 < 0 else ax1
                    ax2 = ax2 + self.event_dim if ax2 < 0 else ax2
                    return jnp.swapaxes(x, ax1, ax2)
                else:  # and finally the cases where the leaf is a scalar (e.g. things like log_partition or base_measure)
                    return x
            return x

        # maps the swap_axes_of_leaf function to all the leaves of the Distribution object, which may include arbitary Pytrees (other distributions, for example)
        data_fields, aux_fields = self.tree_flatten()
        data_fields = jtu.tree_map(lambda x: swap_axes_of_leaf(x, axis1, axis2), data_fields)

        # Now update `batch_shape` to reflect the new ordering of the dimensions in `batch_shape`

        # first we have to correct the axes in case they're negative, in case there's an event dimension
        axis1 = axis1 + self.event_dim if axis1 < 0 else axis1
        axis2 = axis2 + self.event_dim if axis2 < 0 else axis2

        # Convert batch_shape to a list to make it mutable
        batch_shape_list = list(self.batch_shape)

        # Perform the swap
        batch_shape_list[axis1], batch_shape_list[axis2] = (
            batch_shape_list[axis2],
            batch_shape_list[axis1],
        )

        # Convert it back to a tuple and assign it to swapped_dist.batch_shape
        replace = {"batch_shape": tuple(batch_shape_list)}
        swapped_dist = self.tree_unflatten_and_replace(aux_fields, data_fields, replace)
        return swapped_dist

    def moveaxis(self, source: int, destination: int):
        """
        Moves ones batch axis of the component distributions of a Pytree to another position.
        Only supports integer values for source and destination (can only move one axis at a time)
        """

        # convert axes into absolute indices (that account for event_dim), if they are negative (i.e., -1 becomes -3)
        absolute_src = source - self.event_dim if source < 0 else source
        absolute_dest = destination - self.event_dim if destination < 0 else destination

        # add the extra batch dimensions, if A) the leaf is an Array and B) if the Array you're trying to reshape has shape-compatible dimensions
        def moveaxis_of_leaf(x, src_batch, dest_batch, src_abs, dest_abs):
            if isinstance(x, Array):
                if x.ndim == self.batch_dim + self.event_dim:
                    return jnp.moveaxis(x, src_abs, dest_abs)
                elif x.ndim == self.batch_dim:
                    return jnp.moveaxis(x, src_batch, dest_batch)
            return x

        # maps the swap_axes_of_leaf function to all the leaves of the Distribution object, which may include arbitary Pytrees (other distributions, for example)
        data_fields, aux_fields = self.tree_flatten()
        data_fields = jtu.tree_map(
            lambda x: moveaxis_of_leaf(x, source, destination, absolute_src, absolute_dest), data_fields
        )

        # Now update `batch_shape` to reflect the new ordering of the dimensions in `batch_shape`

        source = normalize_axis_tuple(source, self.batch_dim, "source")[0]
        destination = normalize_axis_tuple(destination, self.batch_dim, "destination")[0]

        # Convert batch_shape to a list to make it mutable
        batch_shape_list = [dim for n, dim in enumerate(self.batch_shape) if n != source]

        # Perform the moveaxis
        batch_shape_list.insert(destination, self.batch_shape[source])

        # Convert it back to a tuple and assign it to swapped_dist.batch_shape
        replace = {"batch_shape": tuple(batch_shape_list)}
        moved_dist = self.tree_unflatten_and_replace(aux_fields, data_fields, replace)
        return moved_dist

    def copy(self):
        """
        Returns a copy of the distribution.
        """
        return utils.tree_copy(self)

    def infer_shapes(self, tensor: jnp.ndarray, event_dim: int) -> tuple[int, int]:
        """
        Infers the batch and event shapes of the distribution from the natural parameters.
        """
        batch_shape = tensor.shape[:-event_dim]
        event_shape = tensor.shape[-event_dim:]
        return batch_shape, event_shape

    @classmethod
    def gather_pytree_data_fields(cls):
        """
        Recursively gathers all pytree_data_fields from the class hierarchy.

        Returns:
            A tuple of all pytree_data_fields found in the class hierarchy.
        """
        bases = inspect.getmro(cls)
        all_pytree_data_fields = ()
        for base in bases:
            if issubclass(base, Distribution):
                all_pytree_data_fields += base.__dict__.get("pytree_data_fields", ())
        return tuple(set(all_pytree_data_fields))

    @classmethod
    def gather_pytree_aux_fields(cls):
        """
        Gather all pytree auxiliary fields from the base classes of the given class.

        Args:
            cls: The class to gather pytree auxiliary fields from.

        Returns:
            A tuple of all pytree auxiliary fields from the base classes of the given class.
        """
        bases = inspect.getmro(cls)
        all_pytree_aux_fields = ()
        for base in bases:
            if issubclass(base, Distribution):
                all_pytree_aux_fields += base.__dict__.get("pytree_aux_fields", ())
        return tuple(set(all_pytree_aux_fields))

    def tree_flatten(self):
        """Flattens the distribution into a tuple of data and auxiliary values.

        Returns:
            A tuple containing the data values and auxiliary values of the distribution.
        """
        data_fields = type(self).gather_pytree_data_fields()
        aux_fields = type(self).gather_pytree_aux_fields()

        data_values = tuple(getattr(self, field) for field in data_fields)
        aux_values = tuple(getattr(self, field) for field in aux_fields)

        return data_values, aux_values

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        """
        Reconstructs an instance of the class from its flattened representation.
        Properties like `event_dim` and `batch_dim`, rather than being stored as dynamic datafields in `params` (the
        equivalent of pytree_data_fields), are inferred from the shapes of the data fields `event_shape` and `batch_shape`.

        Args:
            cls: The class to reconstruct.
            aux_data: Auxiliary (aka, static) data needed to reconstruct the instance.
            params: The flattened representation of the instance's dynamic data fields.

        Returns:
            An instance of the class reconstructed from its flattened representation.
        """
        instance = cls.__new__(cls)
        for k, v in zip(cls.gather_pytree_data_fields(), params):
            setattr(instance, k, v)
        for k, v in zip(cls.gather_pytree_aux_fields(), aux_data):
            if k not in ["event_dim", "batch_dim"]:
                setattr(instance, k, v)

        setattr(instance, "event_dim", len(instance.event_shape))
        setattr(instance, "batch_dim", len(instance.batch_shape))
        return instance

    @classmethod
    def tree_unflatten_and_replace(cls, aux_data, params, replace):
        """
        Reconstructs an instance of the class from its flattened representation.
        Properties like `event_dim` and `batch_dim`, rather than being stored as dynamic datafields in `params` (the
        equivalent of pytree_data_fields), are inferred from the shapes of the data fields `event_shape` and `batch_shape`.

        Args:
            cls: The class to reconstruct.
            aux_data: Auxiliary (aka, static) data needed to reconstruct the instance.
            params: The flattened representation of the instance's dynamic data fields.

        Returns:
            An instance of the class reconstructed from its flattened representation.
        """
        instance = cls.__new__(cls)
        for k, v in zip(cls.gather_pytree_data_fields(), params):
            setattr(instance, k, v)
        for k, v in zip(cls.gather_pytree_aux_fields(), aux_data):
            if k not in ["event_dim", "batch_dim"]:
                if k in replace:
                    setattr(instance, k, replace[k])
                else:
                    setattr(instance, k, v)

        setattr(instance, "event_dim", len(instance.event_shape))
        setattr(instance, "batch_dim", len(instance.batch_shape))
        return instance

    def __hash__(self):
        return hash(tuple(jtu.tree_leaves(self)))

    def __eq__(self, other):
        return utils.tree_equal(self, other)


DEFAULT_EVENT_DIM = 1


class Delta(Distribution):
    "Dirac delta distribution"

    pytree_data_fields = ("values", "residual")

    def __init__(self, values: Array, event_dim: Optional[int] = DEFAULT_EVENT_DIM):
        batch_shape, event_shape = values.shape[:-event_dim], values.shape[-event_dim:]
        super().__init__(DEFAULT_EVENT_DIM, batch_shape, event_shape)
        self.values = values
        # TODO: Why does Delta distribution need residual?
        self.residual = jnp.zeros(batch_shape)

    @property
    def p(self) -> jnp.ndarray:
        """
        Returns log probabilities.
        """
        return self.values

    @property
    def mean(self) -> jnp.ndarray:
        """
        Returns the mean parameter of the distribution (aka alias for `p`)
        """
        return self.p

    def expected_x(self):
        return self.p

    def expected_xx(self):
        return self.p * self.p.mT

    def log_partition(self):
        return jnp.zeros(len(self.event_shape) * (1,))

    def entropy(self):
        return jnp.zeros(len(self.event_shape) * (1,))

    def __mul__(self, other):
        """
        Overloads the * operator combine the Delta distribution with another distribution, which by definition is just another instance of the Delta distribution.
        """
        # Check if the other instance is of the same class as self
        if isinstance(other, self.__class__):
            raise ValueError(f"Cannot multiply two Delta messages!")

        return self.copy()
