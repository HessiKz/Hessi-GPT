from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import einsum
from einops import rearrange
from einops import reduce
from einops import repeat
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int


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


# Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced
# transformer with rotary position embedding, 2021.
class _RotaryPositionalEncoding(eqx.Module):
    theta: float
    query_key_dim: int
    max_sequence_len: int

    def __init__(self, theta: float, query_key_dim: int, max_sequence_len: int):
        assert query_key_dim % 2 == 0, "Query/key dimension must be even to apply RoPE"
        self.theta = theta
        self.query_key_dim = query_key_dim
        self.max_sequence_len = max_sequence_len

    def __call__(
        self, x: Float[Array, "sequence query_key"], positions: Int[Array, " sequence"]
    ) -> Float[Array, "sequence query_key"]:
        with jax.ensure_compile_time_eval():
            thetas = jnp.arange(self.max_sequence_len).reshape(-1, 1) / (
                self.theta
                ** (2 * jnp.arange(self.query_key_dim // 2) / self.query_key_dim)
            )
            sines, cosines = jnp.sin(thetas), jnp.cos(thetas)
            blocks = rearrange(
                [cosines, -sines, sines, cosines],
                "(row column) i k -> i k row column",
                row=2,
                column=2,
            )
        rotation_blocks = rearrange(
            blocks[positions], "sequence k row col -> sequence (k row) col"
        )
        x = repeat(x, "sequence (query_key n) -> sequence n (query_key m)", n=2, m=2)
        return einsum(
            x,
            rotation_blocks,
            "sequence col query_key, sequence query_key col -> sequence query_key",
        )


def _scaled_dot_product_attention(
    queries: Float[Array, "queries query_key"],
    keys: Float[Array, "keys query_key"],
    values: Float[Array, "keys value_dim"],
    mask: None | Bool[Array, "queries keys"] = None,
) -> Float[Array, "queries value_dim"]:
    attention_logits = einsum(
        queries,
        keys,
        "queries query_key, keys query_key -> queries keys",
    ) / jnp.sqrt(queries.shape[-1])
    if mask is not None:
        attention_logits = jnp.where(mask, attention_logits, -jnp.inf)
    # Subtract maximum logit in each row for numerical stability
    attention_logits = attention_logits - reduce(
        attention_logits, "queries keys -> queries 1", "max"
    )
    attention_scores = jax.nn.softmax(attention_logits, axis=1)
    return einsum(
        attention_scores,
        values,
        "queries keys, keys value_dim -> queries value_dim",
    )


class _CausalMultiHeadSelfAttention(eqx.Module):
    W_qkv: eqx.nn.Linear
    W_o: eqx.nn.Linear
    num_heads: int
    rope: _RotaryPositionalEncoding | None

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        key: jax.Array,
        rope: _RotaryPositionalEncoding | None = None,
    ):
        self.num_heads = num_heads
        qkv_key, output_key = jax.random.split(key, 2)
        self.rope = rope
        hidden_dim = input_dim // num_heads
        # We calculate queries, keys and values with a single matrix multiply
        self.W_qkv = eqx.nn.Linear(
            in_features=input_dim,
            out_features=3 * num_heads * hidden_dim,
            use_bias=False,
            key=qkv_key,
        )
        self.W_o = eqx.nn.Linear(
            in_features=num_heads * hidden_dim,
            out_features=input_dim,
            use_bias=False,
            key=output_key,
        )

    def __call__(
        self,
        x: Float[Array, "sequence input_dim"],
        positions: Int[Array, " sequence"] | None = None,
    ) -> Float[Array, "sequence input_dim"]:
        seq_len = x.shape[0]
        qkv = jax.vmap(self.W_qkv)(x)
        queries, keys, values = rearrange(
            qkv,
            "sequence (qkv head hidden) -> qkv head sequence hidden",
            qkv=3,
            head=self.num_heads,
        )
        if self.rope is not None:
            rope = jax.vmap(self.rope, in_axes=(0, None))
            if positions is None:
                positions = jnp.arange(seq_len)
            queries = rope(queries, positions)
            keys = rope(keys, positions)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        attention_outputs = jax.vmap(
            _scaled_dot_product_attention, in_axes=(0, 0, 0, None)
        )(queries, keys, values, causal_mask)
        outputs = jax.vmap(self.W_o)(
            rearrange(
                attention_outputs, "head sequence hidden -> sequence (head hidden)"
            )
        )
        return outputs
