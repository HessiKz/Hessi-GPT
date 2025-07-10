import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from llm.transformer import TransformerLanguageModel


def decode_greedy(model: TransformerLanguageModel, context: list[int]) -> int:
    window, index = _pad_context(context, model.max_sequence_len)
    probabilities = _get_token_probabilities(model, window)
    return int(probabilities[index].argmax())


@eqx.filter_jit
def _get_token_probabilities(
    model: TransformerLanguageModel, inputs: Int[Array, " sequence"]
) -> Float[Array, "sequence vocab"]:
    logits = eqx.nn.inference_mode(model)(inputs)
    return jax.nn.softmax(logits)


def _pad_context(
    context: list[int], max_sequence_len: int
) -> tuple[Int[Array, " sequence"], int]:
    window = jnp.array(context[len(context) - max_sequence_len :])
    index = window.size - 1
    return jnp.pad(window, (0, max_sequence_len - window.size)), index
