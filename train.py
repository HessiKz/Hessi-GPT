from pathlib import Path

import hydra
from llm.tokenization import Tokenizer
from omegaconf import DictConfig
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    tokenizer = get_tokenizer(cfg.tokenization)
    del tokenizer


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


if __name__ == "__main__":
    main()
