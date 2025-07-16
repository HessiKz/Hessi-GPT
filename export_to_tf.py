import equinox as eqx
import hydra
import jax
import orbax.checkpoint as ocp
import tensorflow as tf
from jax.experimental import jax2tf
from llm.transformer import TransformerLanguageModel
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    model = eqx.nn.inference_mode(
        load_model_from_checkpoint(checkpointer, cfg.exporting.checkpoint)
    )
    tf_model = convert_to_tf_model(model)
    tf.saved_model.save(tf_model, cfg.exporting.output_path)


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


def convert_to_tf_model(model: TransformerLanguageModel) -> tf.Module:
    tf_model = tf.Module()
    tf_model.f = tf.function(
        jax2tf.convert(model, enable_xla=False, with_gradient=False),
        input_signature=[tf.TensorSpec([model.max_sequence_len], tf.int32)],  # type: ignore
        autograph=False,
    )
    return tf_model


if __name__ == "__main__":
    main()
