from pathlib import Path

import hydra
import jax
from llm.tokenization import Tokenizer
from llm.transformer import TransformerLanguageModel
from omegaconf import DictConfig
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    random_key = jax.random.PRNGKey(cfg.random_seed)
    tokenizer = get_tokenizer(cfg.tokenization)
    random_key, model_key = jax.random.split(random_key)
    model = get_model(cfg.model, model_key)
    del tokenizer, model


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


def get_model(cfg: DictConfig, random_key: jax.Array) -> TransformerLanguageModel:
    lm = TransformerLanguageModel(key=random_key, **cfg)  # type: ignore
    return lm


if __name__ == "__main__":
    main()
