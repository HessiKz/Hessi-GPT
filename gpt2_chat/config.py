# Hessi-GPT — https://github.com/HessiKz/
"""GPT-2 124M chat fine-tune and inference configuration."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints" / "gpt2-chat"
BASE_MODEL = "gpt2"

USER_TAG = "User"
ASSISTANT_TAG = "Assistant"
TURN_END = "\n\n"

BLOCK_SIZE = 256
MAX_TRAIN_SAMPLES = 800
MAX_VAL_SAMPLES = 40

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 3e-5
MAX_STEPS = 60
WARMUP_STEPS = 6
EVAL_EVERY = 60
SAVE_EVERY = 60

GENERATION_KWARGS = {
    "max_new_tokens": 120,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.7,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 3,
}
