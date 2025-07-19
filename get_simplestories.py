from pathlib import Path

import datasets
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

SIMPLESTORIES = "SimpleStories/SimpleStories"
VALIDATION_PROPORTION = 0.05


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    train_split_info = datasets.get_dataset_infos(SIMPLESTORIES)["default"]
    assert train_split_info.splits is not None
    num_examples = train_split_info.splits["train"].num_examples
    num_val_examples = round(num_examples * VALIDATION_PROPORTION)
    separator_token = cfg.tokenization.training.pre_tokenization_split_token
    dataset = datasets.load_dataset(SIMPLESTORIES, streaming=True, split="train")
    train_path, val_path = Path(cfg.train_corpus_path), Path(cfg.val_corpus_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(train_path, "w") as f_train,
        open(val_path, "w") as f_val,
    ):
        desc = f"Saving SimpleStories text into {train_path}, {val_path}"
        for example_idx, example in tqdm(
            enumerate(dataset), desc=desc, total=num_examples
        ):
            story = example["story"]  # type: ignore
            output_file = f_val if example_idx < num_val_examples else f_train
            output_file.write(story)
            output_file.write(separator_token)


if __name__ == "__main__":
    main()
