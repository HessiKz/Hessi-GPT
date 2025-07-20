import os
from pathlib import Path
from typing import Iterator

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import PyTree
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from tqdm import tqdm

from minigpt.decoding import decode_greedy
from minigpt.tokenization import Tokenizer
from minigpt.transformer import TransformerLanguageModel


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    random_key = jax.random.PRNGKey(cfg.random_seed)
    set_xla_flags()
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
    random_key, model_key = jax.random.split(random_key)
    model = get_model(cfg.model, model_key)
    print(f"Model trainable parameters: {model.num_trainable_parameters:_}")
    mp_policy = jmp.get_policy(cfg.training.mixed_precision_policy)
    model = mp_policy.cast_to_param(model)
    optimizer = hydra.utils.instantiate(cfg.training.optimizer)
    with get_checkpoint_manager(cfg.training.checkpoints_path) as checkpoint_mgr:
        train(
            model,
            optimizer,
            train_tokens,
            val_tokens,
            tokenizer,
            mp_policy,
            cfg.training,
            cfg.model,
            cfg.model.max_sequence_len,
            checkpoint_mgr,
            random_key,
        )


def set_xla_flags() -> None:
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=false "
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
    tokenizer: Tokenizer,
    mp_policy: jmp.Policy,
    train_cfg: DictConfig,
    model_cfg: DictConfig,
    max_sequence_len: int,
    checkpoint_manager: ocp.CheckpointManager,
    random_key: jax.Array,
) -> None:
    def loss_fn(
        model: TransformerLanguageModel,
        x: Int[Array, "batch sequence"],
        y: Int[Array, "batch sequence"],
    ) -> Float[Array, ""]:
        y_pred_logits = jax.vmap(model)(x)
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
        model = mp_policy.cast_to_compute(model)
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
                current_key,
                (train_cfg.batch_size,),
                0,
                train_tokens.size - max_sequence_len,
            )
            indices = jnp.arange(max_sequence_len + 1) + start_indices.reshape(-1, 1)
            x, y = train_tokens[indices[:, :-1]], train_tokens[indices[:, 1:]]
            yield x, y

    def val_data_iter() -> Iterator[
        tuple[Int[Array, "batch sequence"], Int[Array, "batch sequence"]]
    ]:
        for i in range(
            0, val_tokens.size, train_cfg.eval.batch_size * max_sequence_len
        ):
            start_indices = jnp.arange(
                i, i + train_cfg.eval.batch_size * max_sequence_len, max_sequence_len
            )
            indices = jnp.arange(max_sequence_len + 1) + start_indices.reshape(-1, 1)
            x, y = train_tokens[indices[:, :-1]], train_tokens[indices[:, 1:]]
            yield x, y

    def generate_text(
        model: TransformerLanguageModel,
    ) -> str:
        context_tokens = tokenizer.encode(train_cfg.eval.generate_text.prompt)
        for _ in range(train_cfg.eval.generate_text.num_tokens):
            token = decode_greedy(model, context_tokens)
            context_tokens.append(token)
        return tokenizer.decode(context_tokens)

    tb_writer = SummaryWriter()
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    opt_state = mp_policy.cast_to_param(opt_state)
    hyperparams = OmegaConf.to_container(model_cfg, resolve=True)
    for step, (x, y) in zip(range(train_cfg.num_steps), train_data_iter()):
        model, opt_state, train_loss = make_train_step(model, opt_state, x, y)
        print(f"Step {step}: Training loss {train_loss}")
        tb_writer.add_scalar("step_loss/training", train_loss, step)
        if step % train_cfg.eval.every_n_steps == 0:
            inference_model = eqx.nn.inference_mode(model)
            val_losses = []
            for x, y in val_data_iter():
                loss = eqx.filter_jit(loss_fn)(inference_model, x, y)
                val_losses.append(loss * x.size)
            mean_val_loss = sum(val_losses) / val_tokens.size
            mean_val_perplexity = jnp.exp(mean_val_loss)
            print(
                f"Step {step}: Validation loss {mean_val_loss}, perplexity {mean_val_perplexity}"
            )
            tb_writer.add_scalar("step_loss/validation", mean_val_loss, step)
            tb_writer.add_scalar(
                "step_perplexity/validation", mean_val_perplexity, step
            )
            text = generate_text(inference_model)
            print(f'Step {step}: generated text (greedy decoding) "{text}"')
            tb_writer.add_text("step_generated_text", text, step)
            checkpoint_manager.save(
                step,
                args=ocp.args.Composite(
                    model=ocp.args.StandardSave(model),  # type: ignore
                    opt_state=ocp.args.StandardSave(opt_state),  # type: ignore
                    metadata=ocp.args.JsonSave(  # type: ignore
                        {"val_loss": float(mean_val_loss), "hyperparams": hyperparams}  # type: ignore
                    ),
                ),
            )
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
