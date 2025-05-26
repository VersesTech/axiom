from collections.abc import Callable, Sequence
from typing import Any, Optional, TYPE_CHECKING, Union

import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, Bool, Float, PyTree, PyTreeDef
from typing import Tuple
from jax.tree_util import register_pytree_node_class
import numpy as np
import jax


@register_pytree_node_class
class ArrayDict:
    def __new__(cls, **kwargs):
        instance = super().__new__(cls)
        # Set '_fields' attribute to store all field names
        object.__setattr__(instance, "_fields", kwargs)
        return instance

    def get(self, key, value=None):
        return self._fields.get(key, value)

    def items(self):
        return self._fields.items()

    def keys(self):
        return self._fields.keys()

    def values(self):
        return self._fields.values()

    def __getitem__(self, key):
        return self._fields[key]

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            return super().__getattribute__(item)
        try:
            return self._fields[item]
        except KeyError as exc:
            raise AttributeError(f"'{self.__class__.__name__}' object has not attribute '{item}'") from exc

    def __setattr__(self, key, value):
        # Prevent modifications to attributes
        raise AttributeError("Cannot modify immutable instance")

    def __delattr__(self, key):
        # Prevent deletions of attributes
        raise AttributeError("Cannot delete attributes of an immutable instance")

    def __repr__(self):
        fields = ", ".join(f"{name}={getattr(self, name)!r}" for name in self._fields)
        return f"{self.__class__.__name__}({fields})"

    def tree_flatten(self):
        values = []
        keys = []
        for key, value in sorted(self._fields.items()):
            values.append(value)
            keys.append(key)

        return values, keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(**dict(zip(keys, values)))


def size(array_dict: PyTree) -> int:
    total_size = 0
    for param in jtu.tree_leaves(array_dict):
        if isinstance(param, jnp.ndarray):
            total_size = total_size + param.size
    return total_size


def sum_pytrees(*pytrees):
    return jtu.tree_map(lambda *args: sum(args), *pytrees)


def zeros_like(array_dict: ArrayDict) -> ArrayDict:
    def zeros_for_value(value):
        if isinstance(value, jnp.ndarray):
            return jnp.zeros(value.shape, value.dtype)
        elif isinstance(value, ArrayDict):
            return zeros_like(value)
        elif isinstance(value, dict):
            return {k: zeros_for_value(v) for k, v in value.items()}
        else:
            return value

    return ArrayDict(**{key: zeros_for_value(val) for key, val in array_dict.items()})


# NOTE: this is not copying any deep structure,
# just re-referencing leaves in a new pytree with same structure
def tree_copy(tree: PyTree) -> PyTree:
    def copy(x):
        return x

    return jtu.tree_map(copy, tree)


def apply_add(dist: PyTree, updates: PyTree) -> PyTree:
    """
    A tree_map-broadcasted addition operation.
    Useful for adding PyTrees together while handling None leaves.
    """

    def _apply_add(u, p):
        if u is None:
            return p
        else:
            return p + u

    def _is_none(x):
        return x is None

    return jtu.tree_map(_apply_add, updates, dist, is_leaf=_is_none)


def apply_scale(dist: PyTree, scale=1.0) -> PyTree:
    """
    A tree_map-broadcasted scale operation.
    Useful for scaling PyTrees while handling None leaves.
    """

    def _apply_scale(leaf):
        if leaf is None:
            return None
        else:
            return leaf * scale

    def _is_none(x):
        return x is None

    return jtu.tree_map(_apply_scale, dist, is_leaf=_is_none)


def tree_marginalize(dist: PyTree, weights: Array, dims: Tuple[int], keepdims=False) -> PyTree:

    def apply_marginalization(leaf, reduce_dims=False):
        if leaf is None:
            return None
        else:
            return (leaf * weights).sum(dims, keepdims=reduce_dims)

    def _is_none(x):
        return x is None

    return jtu.tree_map(lambda x: apply_marginalization(x, reduce_dims=keepdims), dist, is_leaf=_is_none)


def map_and_multiply(a: ArrayDict, b: ArrayDict, sum_dim: int, mapping: dict = None):
    if mapping is not None:
        mapped_b = ArrayDict(**{a_key: b.get(mapping[a_key]) for a_key in a.keys() if mapping[a_key] in b.keys()})
    else:
        mapped_b = b

    def multiply_and_sum(x, y):
        return jnp.sum(x * y, axis=range(-sum_dim, 0), keepdims=True)

    result = jtu.tree_map(multiply_and_sum, a, mapped_b)

    return jtu.tree_reduce(lambda x, y: x + y, result)


def params_to_tx(mapping):
    """
    Decorator to map a PyTree of statistics to a PyTree of natural parameters.
    The mapping is specified as a dictionary, where the keys are the names of the natural parameters,
    and the values are the names of the statistics.
    """

    def decorator(cls):
        cls.params_to_tx = mapping
        return cls

    return decorator


def map_dict_names(params: ArrayDict, name_mapping: dict = None) -> ArrayDict:
    """
    Sets create a new array dict where the named-leaves of the ArrayDict params.k are mapped to leaves in a new ArrayDict,
    with names contained in the key-mapping dictionary `name_mapping[k]`.
    """
    return ArrayDict(**{name_mapping[k]: v for k, v in params.items()})


def tree_equal(
    *pytrees: PyTree,
    typematch: bool = False,
    rtol=0.0,
    atol=0.0,
):
    """
    TODO: take from equinox - rewrite / credit
    """
    flat, treedef = jtu.tree_flatten(pytrees[0])
    traced_out = True
    for pytree in pytrees[1:]:
        flat_, treedef_ = jtu.tree_flatten(pytree)
        if treedef_ != treedef:
            return False
        assert len(flat) == len(flat_)
        for elem, elem_ in zip(flat, flat_):
            if typematch:
                if not isinstance(elem, type(elem_)):
                    return False
            if isinstance(elem, (np.ndarray, np.generic)) and isinstance(elem_, (np.ndarray, np.generic)):
                if (
                    (elem.shape != elem_.shape)
                    or (elem.dtype != elem_.dtype)
                    or not _array_equal(elem, elem_, rtol, atol)
                ):
                    return False
            elif is_array(elem):
                if is_array(elem_):
                    if (elem.shape != elem_.shape) or (elem.dtype != elem_.dtype):
                        return False
                    traced_out = traced_out & _array_equal(elem, elem_, rtol, atol)
                else:
                    return False
            else:
                if is_array(elem_):
                    return False
                else:
                    if elem != elem_:
                        return False
    return traced_out


def is_array(element) -> bool:
    """Returns `True` if `element` is a JAX array or NumPy array."""
    return isinstance(element, (np.ndarray, np.generic, jax.Array))


def _array_equal(x, y, rtol, atol):
    assert x.dtype == y.dtype
    npi = jnp if isinstance(x, Array) or isinstance(y, Array) else np
    if (
        isinstance(rtol, (int, float)) and isinstance(atol, (int, float)) and rtol == 0 and atol == 0
    ) or not npi.issubdtype(x.dtype, npi.inexact):
        return npi.all(x == y)
    else:
        return npi.allclose(x, y, rtol=rtol, atol=atol)


# Copy from equionox package --- Apache 2.0 Licence


class _LeafWrapper:
    def __init__(self, value: Any):
        self.value = value


def _remove_leaf_wrapper(x: _LeafWrapper) -> Any:
    if not isinstance(x, _LeafWrapper):
        raise TypeError(f"Operation undefined, {x} is not a leaf of the pytree.")
    return x.value


class _CountedIdDict:
    def __init__(self, keys, values):
        assert len(keys) == len(values)
        self._dict = {id(k): v for k, v in zip(keys, values)}
        self._count = {id(k): 0 for k in keys}

    def __contains__(self, item):
        return id(item) in self._dict

    def __getitem__(self, item):
        self._count[id(item)] += 1
        return self._dict[id(item)]

    def get(self, item, default):
        try:
            return self[item]
        except KeyError:
            return default

    def count(self, item):
        return self._count[id(item)]


def tree_at(
    where: Callable[[PyTree], Union[Any, Sequence[Any]]],
    pytree: PyTree,
    replace: Union[Any, Sequence[Any]] = None,
    replace_fn: Callable[[Any], Any] = None,
    is_leaf: Optional[Callable[[Any], bool]] = None,
):
    """Modifies a leaf or subtree of a PyTree. (A bit like using `.at[].set()` on a JAX
    array.)

    The modified PyTree is returned and the original input is left unchanged. Make sure
    to use the return value from this function!

    **Arguments:**

    - `where`: A callable `PyTree -> Node` or `PyTree -> tuple[Node, ...]`. It should
        consume a PyTree with the same structure as `pytree`, and return the node or
        nodes that should be replaced. For example
        `where = lambda mlp: mlp.layers[-1].linear.weight`.
    - `pytree`: The PyTree to modify.
    - `replace`: Either a single element, or a sequence of the same length as returned
        by `where`. This specifies the replacements to make at the locations specified
        by `where`. Mutually exclusive with `replace_fn`.
    - `replace_fn`: A function `Node -> Any`. It will be called on every node specified
        by `where`. The return value from `replace_fn` will be used in its place.
        Mutually exclusive with `replace`.
    - `is_leaf`: As `jtu.tree_flatten`. For example pass `is_leaf=lambda x: x is None`
        to be able to replace `None` values using `tree_at`.

    Note that `where` should not depend on the type of any of the leaves of the
    pytree, e.g. given `pytree = [1, 2, object(), 3]`, then
    `where = lambda x: tuple(xi for xi in x if type(xi) is int)` is not allowed. If you
    really need this behaviour then this example could instead be expressed as
    `where = lambda x: tuple(xi for xi, yi in zip(x, pytree) if type(yi) is int)`.

    **Returns:**

    A copy of the input PyTree, with the appropriate modifications.

    !!! Example

        ```python
        # Here is a pytree
        tree = [1, [2, {"a": 3, "b": 4}]]
        new_leaf = 5
        get_leaf = lambda t: t[1][1]["a"]
        new_tree = eqx.tree_at(get_leaf, tree, 5)
        # new_tree is [1, [2, {"a": 5, "b": 4}]]
        # The original tree is unchanged.
        ```

    !!! Example

        This is useful for performing model surgery. For example:
        ```python
        mlp = eqx.nn.MLP(...)
        new_linear = eqx.nn.Linear(...)
        get_last_layer = lambda m: m.layers[-1]
        new_mlp = eqx.tree_at(get_last_layer, mlp, new_linear)
        ```
        See also the [Tricks](../../tricks) page.
    """  # noqa: E501

    # We need to specify a particular node in a PyTree.
    # This is surprisingly difficult to do! As far as I can see, pretty much the only
    # way of doing this is to specify e.g. `x.foo[0].bar` via `is`, and then pulling
    # a few tricks to try and ensure that the same object doesn't appear multiple
    # times in the same PyTree.
    #
    # So this first `tree_map` serves a dual purpose.
    # 1) Makes a copy of the composite nodes in the PyTree, to avoid aliasing via
    #    e.g. `pytree=[(1,)] * 5`. This has the tuple `(1,)` appear multiple times.
    # 2) It makes each leaf be a unique Python object, as it's wrapped in
    #    `_LeafWrapper`. This is needed because Python caches a few builtin objects:
    #    `assert 0 + 1 is 1`. I think only a few leaf types are subject to this.
    # So point 1) should ensure that all composite nodes are unique Python objects,
    # and point 2) should ensure that all leaves are unique Python objects.
    # Between them, all nodes of `pytree` are handled.
    #
    # I think pretty much the only way this can fail is when using a custom node with
    # singleton-like flatten+unflatten behaviour, which is pretty edge case. And we've
    # added a check for it at the bottom of this function, just to be sure.
    #
    # Whilst we're here: we also double-check that `where` is well-formed and doesn't
    # use leaf information. (As else `node_or_nodes` will be wrong.)
    node_or_nodes_nowrapper = where(pytree)
    pytree = jtu.tree_map(_LeafWrapper, pytree, is_leaf=is_leaf)
    node_or_nodes = where(pytree)
    leaves1, structure1 = jtu.tree_flatten(node_or_nodes_nowrapper, is_leaf=is_leaf)
    leaves2, structure2 = jtu.tree_flatten(node_or_nodes)
    leaves2 = [_remove_leaf_wrapper(x) for x in leaves2]
    if (
        structure1 != structure2
        or len(leaves1) != len(leaves2)
        or any(l1 is not l2 for l1, l2 in zip(leaves1, leaves2))
    ):
        raise ValueError(
            "`where` must use just the PyTree structure of `pytree`. `where` must not "
            "depend on the leaves in `pytree`."
        )
    del node_or_nodes_nowrapper, leaves1, structure1, leaves2, structure2

    # Normalise whether we were passed a single node or a sequence of nodes.
    in_pytree = False

    def _in_pytree(x):
        nonlocal in_pytree
        if x is node_or_nodes:  # noqa: F821
            in_pytree = True
        return x  # needed for jax.tree_util.Partial, which has a dodgy constructor

    jtu.tree_map(_in_pytree, pytree, is_leaf=lambda x: x is node_or_nodes)  # noqa: F821
    if in_pytree:
        nodes = (node_or_nodes,)
        if replace is not None:
            replace = (replace,)
    else:
        nodes = node_or_nodes
    del in_pytree, node_or_nodes

    # Normalise replace vs replace_fn
    if replace is None:
        if replace_fn is None:
            raise ValueError("Precisely one of `replace` and `replace_fn` must be specified.")
        else:

            def _replace_fn(x):
                x = jtu.tree_map(_remove_leaf_wrapper, x)
                return replace_fn(x)

            replace_fns = [_replace_fn] * len(nodes)
    else:
        if replace_fn is None:
            if len(nodes) != len(replace):
                raise ValueError("`where` must return a sequence of leaves of the same length as " "`replace`.")
            replace_fns = [lambda _, r=r: r for r in replace]
        else:
            raise ValueError("Precisely one of `replace` and `replace_fn` must be specified.")
    node_replace_fns = _CountedIdDict(nodes, replace_fns)

    # Actually do the replacement
    def _make_replacement(x: Any) -> Any:
        return node_replace_fns.get(x, _remove_leaf_wrapper)(x)

    out = jtu.tree_map(_make_replacement, pytree, is_leaf=lambda x: x in node_replace_fns)

    # Check that `where` is well-formed.
    for node in nodes:
        count = node_replace_fns.count(node)
        if count == 0:
            raise ValueError("`where` does not specify an element or elements of `pytree`.")
        elif count == 1:
            pass
        else:
            raise ValueError(
                "`where` does not uniquely identify a single element of `pytree`. This "
                "usually occurs when trying to replace a `None` value:\n"
                "\n"
                "  >>> eqx.tree_at(lambda t: t[0], (None, None, 1), True)\n"
                "\n"
                "\n"
                "for which the fix is to specify that `None`s should be treated as "
                "leaves:\n"
                "\n"
                "  >>> eqx.tree_at(lambda t: t[0], (None, None, 1), True,\n"
                "  ...             is_leaf=lambda x: x is None)"
            )

    return out
