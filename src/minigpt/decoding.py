import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from minigpt.transformer import TransformerLanguageModel


def decode_greedy(model: TransformerLanguageModel, context: list[int]) -> int:
    window, index = _pad_context(context, model.max_sequence_len)
    probabilities = _get_token_probabilities(model, window)
    return int(probabilities[index].argmax())


def decode_nucleus(
    model: TransformerLanguageModel,
    context: list[int],
    key: jax.Array,
    p: float = 0.5,
    temperature: float = 0.1,
) -> int:
    _check_temperature(temperature)
    if p < 0 or p > 1:
        raise ValueError("p must be strictly between 0 and 1")
    window, index = _pad_context(context, model.max_sequence_len)
    probabilities = _get_token_probabilities(model, window, temperature)
    return int(_sample_nucleus(probabilities[index], p, key))


def decode_topk(
    model: TransformerLanguageModel,
    context: list[int],
    key: jax.Array,
    k: int = 50,
    temperature: float = 0.1,
) -> int:
    _check_temperature(temperature)
    vocab_size = model.embedding.num_embeddings
    if k <= 0 or k > vocab_size:
        raise ValueError(
            f"k must be greater than 0 and not greater than the model's vocabulary size ({vocab_size})"
        )
    window, index = _pad_context(context, model.max_sequence_len)
    probabilities = _get_token_probabilities(model, window, temperature)
    return int(_sample_topk(probabilities[index], k, key))


@eqx.filter_jit
def _get_token_probabilities(
    model: TransformerLanguageModel,
    inputs: Int[Array, " sequence"],
    temperature: float = 1,
) -> Float[Array, "sequence vocab"]:
    logits = eqx.nn.inference_mode(model)(inputs)
    return jax.nn.softmax(logits / temperature)


def _pad_context(
    context: list[int], max_sequence_len: int
) -> tuple[Int[Array, " sequence"], int]:
    window = jnp.array(context[len(context) - max_sequence_len :])
    index = window.size - 1
    return jnp.pad(window, (0, max_sequence_len - window.size)), index


def _check_temperature(temperature: float) -> None:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")


@eqx.filter_jit
def _sample_nucleus(
    probabilities: Float[Array, " vocab"], p: float, key: jax.Array
) -> Int[Array, ""]:
    argsorted_probs = jnp.argsort(probabilities, descending=True)
    sorted_probs = probabilities[argsorted_probs]
    mask = jnp.cumsum(sorted_probs) < p
    mask = jnp.concatenate((jnp.array([1]), mask[:-1]))
    probs = jnp.where(mask, sorted_probs, 0)
    probs = probs / probs.sum()
    return jax.random.choice(key, argsorted_probs, p=probs)


@eqx.filter_jit
def _sample_topk(
    probabilities: Float[Array, " vocab"], k: int, key: jax.Array
) -> Int[Array, ""]:
    argsorted_probs = jnp.argsort(probabilities, descending=True)
    sorted_probs = probabilities[argsorted_probs]
    sorted_probs = sorted_probs.at[k:].set(0)
    probs = sorted_probs / sorted_probs.sum()
    return jax.random.choice(key, argsorted_probs, p=probs)
