from __future__ import annotations

import equinox as eqx
import jax
from jaxtyping import Array
from jaxtyping import Float


class _PositionwiseFFN(eqx.Module):
    linear_1: eqx.nn.Linear
    linear_2: eqx.nn.Linear
    linear_3: eqx.nn.Linear

    def __init__(self, input_output_dim: int, feed_forward_dim: int, key: jax.Array):
        key1, key2, key3 = jax.random.split(key, 3)
        self.linear_1 = eqx.nn.Linear(
            in_features=input_output_dim,
            out_features=feed_forward_dim,
            use_bias=False,
            key=key1,
        )
        self.linear_2 = eqx.nn.Linear(
            in_features=feed_forward_dim,
            out_features=input_output_dim,
            use_bias=False,
            key=key2,
        )
        self.linear_3 = eqx.nn.Linear(
            in_features=input_output_dim,
            out_features=feed_forward_dim,
            use_bias=False,
            key=key3,
        )

    def __call__(self, x: Float[Array, " hidden"]) -> Float[Array, " hidden"]:
        glu = jax.nn.swish(self.linear_1(x)) * self.linear_3(x)
        return self.linear_2(glu)
