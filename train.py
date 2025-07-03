from pathlib import Path

import hydra
import jax
import numpy as np
from llm.tokenization import Tokenizer
from llm.transformer import TransformerLanguageModel
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tqdm import tqdm


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    random_key = jax.random.PRNGKey(cfg.random_seed)
    tokenizer = get_tokenizer(cfg.tokenization)
    train_tokens = get_tokens(
        tokenizer, cfg.train_corpus_path, cfg.tokenization.tokenized_train_set_path
    )
    val_tokens = get_tokens(
        tokenizer, cfg.val_corpus_path, cfg.tokenization.tokenized_val_set_path
    )
    random_key, model_key = jax.random.split(random_key)
    model = get_model(cfg.model, model_key)
    del tokenizer, model, train_tokens, val_tokens


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
        for i, token in tqdm(
            enumerate(tokenizer.encode_iterable(corpus_fd)),
            desc=f"Tokenizing {corpus_path} to {tokens_path}",
        ):
            tokens_fd.write(token.to_bytes(2, byteorder="little"))
    return np.memmap(tokens_output_path, dtype=np.uint16, mode="r")


def get_model(cfg: DictConfig, random_key: jax.Array) -> TransformerLanguageModel:
    lm = TransformerLanguageModel(key=random_key, **cfg)  # type: ignore
    return lm


if __name__ == "__main__":
    main()
