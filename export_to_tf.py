# Hessi-GPT — https://github.com/HessiKz/
import json

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import tensorflow as tf
from jax.experimental import jax2tf
from omegaconf import DictConfig

from minigpt.tokenization import BPETokenizer
from minigpt.transformer import TransformerLanguageModel
from train import get_tokenizer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    model = eqx.nn.inference_mode(
        load_model_from_checkpoint(checkpointer, cfg.exporting.checkpoint)
    )
    # bfloat16 is not supported in e.g. TFJS, so we convert all parameters to float32
    model = ensure_float32(model)
    tf_model = convert_to_tf_model(model)
    tf.saved_model.save(tf_model, cfg.exporting.output_path)
    tokenizer = get_tokenizer(cfg.tokenization)
    assert isinstance(tokenizer, BPETokenizer)
    with open(cfg.exporting.tokenizer_output_path, "w") as f:
        f.write(tokenizer_to_json(tokenizer))


def load_model_from_checkpoint(
    checkpointer: ocp.Checkpointer, checkpoint_dir: str
) -> TransformerLanguageModel:
    restored = checkpointer.restore(checkpoint_dir)
    model_params, hyperparams = restored.model, restored.metadata["hyperparams"]
    paths, _ = jax.tree.flatten_with_path(model_params)
    paths = {jax.tree_util.keystr(path): val for path, val in paths}

    def convert_path(path: tuple, value: jax.Array) -> jax.Array:
        converted_path = tuple(
            (
                jax.tree_util.DictKey(x.name)
                if isinstance(x, jax.tree_util.GetAttrKey)
                else x
            )
            for x in path
        )
        return paths.get(jax.tree_util.keystr(converted_path), value)

    model = TransformerLanguageModel(key=jax.random.PRNGKey(0), **hyperparams)
    model = jax.tree.map_with_path(convert_path, model)
    return model


def ensure_float32(model: TransformerLanguageModel) -> TransformerLanguageModel:
    return jax.tree.map(
        lambda x: x.astype(jnp.float32) if isinstance(x, jax.Array) else x, model
    )


def convert_to_tf_model(model: TransformerLanguageModel) -> tf.Module:
    tf_model = tf.Module()
    tf_model.f = tf.function(
        jax2tf.convert(model, enable_xla=False, with_gradient=False),
        input_signature=[tf.TensorSpec([model.max_sequence_len], tf.int32)],  # type: ignore
        autograph=False,
    )
    return tf_model


def tokenizer_to_json(tokenizer: BPETokenizer) -> str:
    def bytes_to_str(b: bytes) -> str:
        return str(b)[2:-1]  # Escape non-printable characters as \xXX

    tokenizer_dict = {
        "pre_tokenization_regex": tokenizer.pre_tokenization_regex,
        "special_tokens": tokenizer.special_tokens,
        "vocabulary": {
            token_id: bytes_to_str(token)
            for token_id, token in tokenizer.vocabulary.items()
        },
        "merges": [(bytes_to_str(a), bytes_to_str(b)) for a, b in tokenizer.merges],
    }
    return json.dumps(tokenizer_dict)


if __name__ == "__main__":
    main()
