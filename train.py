from pathlib import Path
from typing import Iterator

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from einops import reduce
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import PyTree
from llm.tokenization import Tokenizer
from llm.transformer import TransformerLanguageModel
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from tqdm import tqdm


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    random_key = jax.random.PRNGKey(cfg.random_seed)
    tokenizer = get_tokenizer(cfg.tokenization)
    print(
        f"Loading training token data from {cfg.tokenization.tokenized_train_set_path}"
    )
    train_tokens = get_tokens(
        tokenizer, cfg.train_corpus_path, cfg.tokenization.tokenized_train_set_path
    )
    print(
        f"Loading validation token data from {cfg.tokenization.tokenized_val_set_path}"
    )
    val_tokens = get_tokens(
        tokenizer, cfg.val_corpus_path, cfg.tokenization.tokenized_val_set_path
    )
    del tokenizer
    random_key, model_key = jax.random.split(random_key)
    model = get_model(cfg.model, model_key)
    print(f"Model trainable parameters: {model.num_trainable_parameters:_}")
    optimizer = hydra.utils.instantiate(cfg.training.optimizer)
    with get_checkpoint_manager(cfg.training.checkpoints_path) as checkpoint_mgr:
        train(
            model,
            optimizer,
            train_tokens,
            val_tokens,
            cfg.training,
            cfg.model.max_sequence_len,
            checkpoint_mgr,
            random_key,
        )


def get_tokenizer(cfg: DictConfig) -> Tokenizer:
    save_path = Path(cfg.save_path)
    if save_path.exists():
        print(f"Loading tokenizer from {save_path}")
        tokenizer = Tokenizer.load_from_file(save_path)
        return tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    assert isinstance(tokenizer, Tokenizer)
    print("Training tokenizer:")
    print(OmegaConf.to_yaml(cfg.training, resolve=True))
    tokenizer.train_on_corpus(**cfg.training)
    print(f"Saving tokenizer to {save_path}")
    tokenizer.save_to_file(save_path)
    return tokenizer


def get_tokens(tokenizer: Tokenizer, corpus_path: str, tokens_path: str) -> np.memmap:
    tokens_output_path = Path(tokens_path)
    if tokens_output_path.exists():
        return np.memmap(tokens_output_path, dtype=np.uint16, mode="r")
    tokens_output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(corpus_path) as corpus_fd, open(tokens_output_path, "wb") as tokens_fd:
        for token in tqdm(
            tokenizer.encode_iterable(corpus_fd),
            desc=f"Tokenizing {corpus_path} to {tokens_path}",
        ):
            tokens_fd.write(token.to_bytes(2, byteorder="little"))
    return np.memmap(tokens_output_path, dtype=np.uint16, mode="r")


def get_model(cfg: DictConfig, random_key: jax.Array) -> TransformerLanguageModel:
    lm = TransformerLanguageModel(key=random_key, **cfg)  # type: ignore
    return lm


def get_checkpoint_manager(checkpoints_path: str) -> ocp.CheckpointManager:
    path = Path(checkpoints_path)
    path.mkdir(exist_ok=True, parents=True)
    return ocp.CheckpointManager(path)


def train(
    model: TransformerLanguageModel,
    optimizer: optax.GradientTransformation,
    train_tokens: np.memmap,
    val_tokens: np.memmap,
    cfg: DictConfig,
    max_sequence_len: int,
    checkpoint_manager: ocp.CheckpointManager,
    random_key: jax.Array,
) -> None:
    @eqx.filter_jit
    def loss_fn(
        model: TransformerLanguageModel,
        x: Int[Array, "batch sequence"],
        y: Int[Array, "batch sequence"],
    ) -> Float[Array, ""]:
        y_pred_logits = jax.vmap(model)(x)
        # Subtract maximum logit for numerical stability
        y_pred_logits = y_pred_logits - reduce(
            y_pred_logits, "batch sequence vocab -> batch sequence 1", "max"
        )
        losses = jax.vmap(optax.losses.softmax_cross_entropy_with_integer_labels)(
            y_pred_logits, y
        )
        return jnp.mean(losses)

    @eqx.filter_jit
    def make_train_step(
        model: TransformerLanguageModel,
        opt_state: PyTree,
        x: Int[Array, "batch sequence"],
        y: Int[Array, "batch sequence"],
    ) -> tuple[TransformerLanguageModel, PyTree, Float[Array, ""]]:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def train_data_iter() -> Iterator[
        tuple[Int[Array, "batch sequence"], Int[Array, "batch sequence"]]
    ]:
        _, sampling_key = jax.random.split(random_key)
        while True:
            sampling_key, current_key = jax.random.split(sampling_key)
            start_indices = jax.random.randint(
                current_key, (cfg.batch_size,), 0, train_tokens.size - max_sequence_len
            )
            indices = jnp.arange(max_sequence_len + 1) + start_indices.reshape(-1, 1)
            x, y = train_tokens[indices[:, :-1]], train_tokens[indices[:, 1:]]
            yield x, y

    def val_data_iter() -> Iterator[
        tuple[Int[Array, "batch sequence"], Int[Array, "batch sequence"]]
    ]:
        for i in range(0, val_tokens.size, cfg.eval_batch_size * max_sequence_len):
            start_indices = jnp.arange(
                i, i + cfg.eval_batch_size * max_sequence_len, max_sequence_len
            )
            indices = jnp.arange(max_sequence_len + 1) + start_indices.reshape(-1, 1)
            x, y = train_tokens[indices[:, :-1]], train_tokens[indices[:, 1:]]
            yield x, y

    tb_writer = SummaryWriter()
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    for step, (x, y) in zip(range(cfg.num_steps), train_data_iter()):
        model, opt_state, train_loss = make_train_step(model, opt_state, x, y)
        tb_writer.add_scalar("step_loss/training", train_loss, step)
        if step % cfg.eval_every_n_steps == 0:
            inference_model = eqx.nn.inference_mode(model)
            val_losses = []
            for x, y in val_data_iter():
                loss = loss_fn(inference_model, x, y)
                val_losses.append(loss * x.size)
            mean_val_loss = sum(val_losses) / val_tokens.size
            tb_writer.add_scalar("step_loss/validation", mean_val_loss, step)
            checkpoint_manager.save(
                step,
                args=ocp.args.Composite(
                    model=ocp.args.StandardSave(model),  # type: ignore
                    opt_state=ocp.args.StandardSave(opt_state),  # type: ignore
                    metadata=ocp.args.JsonSave({"val_loss": float(mean_val_loss)}),  # type: ignore
                ),
            )
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
