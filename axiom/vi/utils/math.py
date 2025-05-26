from typing import Optional, Tuple, Union
from jaxtyping import Array
from functools import partial
from jax import lax, vmap, jit
import jax.numpy as jnp
import jax.scipy.special as jsp
import jax.scipy.linalg as linalg
from jax.numpy import expand_dims as expand

from jax.scipy.special import logsumexp
from jax.nn import softmax


# @partial(jit, static_argnames=["return_inverse", "return_logdet"])
# def inv_and_logdet(
#     pos_def_matrix: Array, return_inverse: Optional[bool] = True, return_logdet: Optional[bool] = True
# ) -> Union[float, Tuple[Array, float]]:
#     """
#     Compute the log-determinant of a positive definite matrix and
#     compute a matrix inverse using qr decomposition.
#     """
#     # ensure that logdet is returned when matrix inverse is not computed
#     return_logdet = True if return_inverse == False else return_logdet
#     q, r = jnp.linalg.qr(pos_def_matrix)
#     d = jnp.diagonal(r, axis1=-1, axis2=-2)
#     if return_inverse:
#         matrix_inverse = linalg.solve_triangular(r, q.mT)
#         if return_logdet:
#             logdet = jnp.expand_dims(jnp.log(jnp.abs(d)).sum(-1), (-1, -2))
#             return matrix_inverse, logdet

#         return matrix_inverse

#     logdet = jnp.expand_dims(jnp.log(jnp.abs(d)).sum(-1), (-1, -2))
#     return logdet


@partial(jit, static_argnames=["return_inverse", "return_logdet"])
def inv_and_logdet(
    pos_def_matrix: Array,
    return_inverse: Optional[bool] = True,
    return_logdet: Optional[bool] = True,
) -> Union[float, Tuple[Array, float]]:
    """compute log-determinant of a positive definite matrix and
    compute an matrix inverse using cholesky decomposition.
    """
    # ensure that logdet is returned when matrix inverse is not computed
    return_logdet = True if return_inverse == False else return_logdet

    shape = pos_def_matrix.shape
    # ensure that the matrix is positive definite
    min_eig = jnp.clip(jnp.min(jnp.linalg.eigvalsh(pos_def_matrix)), max=0.0)
    eps = jnp.finfo(pos_def_matrix.dtype).eps
    pos_def_matrix = pos_def_matrix + jnp.broadcast_to(jnp.eye(shape[-1]), shape) * 2 * (eps - min_eig) * (min_eig < 0)

    chol = linalg.cho_factor(pos_def_matrix, lower=True)
    if return_inverse:
        identity = jnp.broadcast_to(jnp.eye(shape[-1]), shape)
        matrix_inverse = linalg.cho_solve(chol, identity)

        if return_logdet:
            logdet = jnp.expand_dims(2 * jnp.log(jnp.diagonal(chol[0], axis1=-1, axis2=-2)).sum(-1), (-1, -2))
            return matrix_inverse, logdet

        return matrix_inverse

    logdet = jnp.expand_dims(2 * jnp.log(jnp.diagonal(chol[0], axis1=-1, axis2=-2)).sum(-1), (-1, -2))
    return logdet


def bdot(x, y):
    """Batched dot product using vmap"""
    assert x.ndim > 1
    assert y.ndim > 1
    shape = jnp.broadcast_shapes(x.shape[:-2], y.shape[:-2])
    x = jnp.broadcast_to(x, shape + x.shape[-2:])
    y = jnp.broadcast_to(y, shape + y.shape[-2:])
    z = vmap(jnp.dot)(x.reshape((-1,) + x.shape[-2:]), y.reshape((-1,) + y.shape[-2:]))
    x_dim = x.shape[-2]
    y_dim = y.shape[-1]
    return z.reshape(
        shape
        + (
            x_dim,
            y_dim,
        )
    )


def positive_leading_eigenvalues(x, iters=10):
    """Checks if all eigenvalues of an array are positive and if not, repeatedly adds an epsilon value to
    the diagonal to attempt to bring them in line. Assumes the input array is square in the last two dimesions.
    """
    eps = jnp.eye(x.shape[-1]) * 0.001
    for i in range(iters):
        eigs = jnp.linalg.eigh(x).eigenvalues
        if jnp.any(eigs <= 0):
            x += eps
        else:
            return x
    raise ValueError("Unable to maintain non-negative leading diagonal, please check your model specification")


def symmetrise(x):
    """Forces a matrix to be symmetric"""
    return (x + x.mT) / 2


def make_posdef(x):
    """Makes a matrix symmetric and attempts to force positive eigenvalues in order to ensure positive definiteness"""
    return positive_leading_eigenvalues(symmetrise(x))


def mvdigamma(x: jnp.ndarray, d: int) -> jnp.ndarray:
    """
    Compute the the multivariate digamma function.
    """
    return jsp.digamma(jnp.expand_dims(x, -1) - jnp.arange(d) / 2.0).sum(-1)


def mvgammaln(x: jnp.ndarray, d: int) -> jnp.ndarray:
    """
    Compute the log of the multivariate gamma function.
    """
    return jnp.sum(lax.lgamma(expand(x, -1) - jnp.arange(d) / 2.0), -1) + d * (d - 1) / 4.0 * jnp.log(jnp.pi)


def stable_logsumexp(x: jnp.ndarray, dims: tuple, keepdims=False) -> jnp.ndarray:
    """
    Compute the logsumexp of x along the dimensions dims.
    """
    return logsumexp(x, axis=dims, keepdims=keepdims)


def stable_softmax(x: jnp.ndarray, dims: tuple) -> jnp.ndarray:
    """
    Compute the softmax of x along the dimensions dims.
    """
    return softmax(x, axis=dims)


def assign_unused(assignments, d_alpha, elbo_contrib, threshold: float = 1.0, fill_prob: float = 10.0):
    """
    Re-assign data-points with low ELBO to unused clusters.

    Args:
        assignments: (num_samples, num_classes) The assignments of the data to the mixture components.
        d_alpha: (num_classes, ) The difference between the prior and the posterior Dirichlet parameters.
        elbo_contrib: (num_samples, ) The ELBO contributions of the assignments.
        threshold: float The threshold for the mixing components to be considered unused.
        fill_prob: float The value to use to fill out the new posterior over assignment probabilities

    Returns:
        The filled assignments.
    """
    unfilled_cluster_idx = d_alpha < threshold  # this is a Boolean mask of which clusters are unfilled
    sorted_elbo_idx = jnp.argsort(elbo_contrib)

    num_to_fill = unfilled_cluster_idx.sum()  # total number of unfilled clusters
    reassign_mask = jnp.arange(assignments.shape[0]) < num_to_fill

    assignments_sorted_by_elbo = assignments[
        sorted_elbo_idx
    ]  # assignments stacked where lowest ELBO assignmenst are first rows

    onehots_base = fill_prob * jnp.eye(d_alpha.shape[0])  # num_classes, num_classes
    # onehots_to_keep = onehots_base[unfilled_cluster_idx] # this breaks vmap
    onehots_to_keep = onehots_base * unfilled_cluster_idx[None, ...]
    onehots_to_keep = jnp.take_along_axis(
        onehots_to_keep, jnp.argsort(onehots_to_keep.sum(-1))[::-1][..., None], axis=0
    )

    sorted_ass_reassigned = (
        assignments_sorted_by_elbo * (1.0 - reassign_mask[..., None])
        + (reassign_mask[..., None]) * onehots_to_keep[jnp.cumsum(reassign_mask) - 1]
    )

    return sorted_ass_reassigned[jnp.argsort(sorted_elbo_idx)]  # sort back into their original order in `assignments`
