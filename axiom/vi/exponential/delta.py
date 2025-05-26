import jax.numpy as jnp
from jaxtyping import Array

from axiom.vi import Distribution, ArrayDict
from typing import Optional


class Delta(Distribution):
    pytree_data_fields = ("p", "posterior_params", "prior_params")

    def __init__(self, batch_shape: tuple, event_shape: tuple, p: Optional[ArrayDict] = None, **parent_kwargs):
        # Allows for using deltas as distributions
        if p is not None:
            self.p = p.p
        else:
            self.p = None

        # TODO: im using this a "conjugate" so need these fields for generic functions
        self.posterior_params = None
        self.prior_params = None

        super().__init__(0, batch_shape, event_shape)

    def expected_log_likelihood(self, x: Array) -> Array:
        # Evaluates to 1 if we have p and x is at the point
        if self.p is not None:
            if jnp.array_equal(x, self.p):
                return jnp.zeros(x.shape)
            else:
                return jnp.log(1e-8) * jnp.ones(x.shape)
        # Otherwise
        return jnp.log(x + 1e-8)

    def update_from_data(self, x: Array, weights: Array, beta: float = 0.0, lr: float = 1.0):
        pass

    def update_from_probabilities(self, x: Array, weights: Array):
        pass

    def mean(self):
        return self.p

    def expected_statistics(self):
        return ArrayDict(p=self.p)

    def entropy(self):
        return 0

    def sample(self):
        return self.p
