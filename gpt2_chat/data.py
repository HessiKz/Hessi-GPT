# Hessi-GPT — https://github.com/HessiKz/
"""Load and tokenize chat datasets for GPT-2 fine-tuning."""

from __future__ import annotations

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from gpt2_chat.config import BLOCK_SIZE, MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES
from gpt2_chat.formatting import format_turn

SEED_DIALOGS: list[tuple[str, str]] = [
    ("Hello!", "Hello! How can I help you today?"),
    ("Hi there!", "Hi! Nice to meet you. What would you like to talk about?"),
    ("How are you?", "I'm doing well, thank you for asking! How are you?"),
    ("What's your name?", "I'm Hessi-GPT, a small chat language model built by HessiKz."),
    ("Who made you?", "I was built by HessiKz as part of the Hessi-GPT project."),
    ("Tell me a joke.", "Why did the developer go broke? Because they used up all their cache."),
    ("What is 2+2?", "2 + 2 equals 4."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Good morning!", "Good morning! Hope you have a great day."),
    ("Thanks!", "You're welcome! Let me know if you need anything else."),
    ("Thank you for your help.", "You're welcome! Happy to help anytime."),
    ("Bye!", "Goodbye! It was nice chatting with you."),
    ("Can you help me?", "Of course! Tell me what you need and I'll do my best."),
    ("What can you do?", "I can chat, answer simple questions, and try to be helpful."),
    ("Nice to meet you.", "Nice to meet you too! Feel free to ask me anything."),
]


def _seed_rows(repeat: int = 30) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _ in range(repeat):
        for user, assistant in SEED_DIALOGS:
            rows.append({"user": user, "assistant": assistant})
    return rows


def _dailydialog_rows(limit: int) -> list[dict[str, str]]:
    try:
        ds = load_dataset("daily_dialog", trust_remote_code=True)
    except Exception as exc:
        print(f"DailyDialog unavailable: {exc}")
        return []
    rows: list[dict[str, str]] = []
    for split in ("train", "validation"):
        if split not in ds:
            continue
        for item in ds[split]:
            dialog = item.get("dialog") or []
            for i in range(0, len(dialog) - 1, 2):
                user = str(dialog[i]).strip()
                assistant = str(dialog[i + 1]).strip()
                if not user or not assistant:
                    continue
                if len(assistant) > 200:
                    continue
                rows.append({"user": user, "assistant": assistant})
                if len(rows) >= limit:
                    return rows
    return rows


def load_chat_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = _seed_rows(repeat=30)
    rows.extend(_dailydialog_rows(MAX_TRAIN_SAMPLES))
    rows = rows[: MAX_TRAIN_SAMPLES + MAX_VAL_SAMPLES]
    split = min(MAX_VAL_SAMPLES, max(1, len(rows) // 20))
    return rows[split:], rows[:split]


def build_text_dataset(tokenizer: PreTrainedTokenizerBase) -> tuple[Dataset, Dataset]:
    train_rows, val_rows = load_chat_rows()
    print(f"Dataset mix: {len(train_rows)} train / {len(val_rows)} val (conversational only)")

    def to_text(batch: dict[str, list[str]]) -> dict[str, list[str]]:
        return {
            "text": [
                format_turn(user, assistant)
                for user, assistant in zip(batch["user"], batch["assistant"])
            ]
        }

    train_ds = Dataset.from_list(train_rows).map(to_text, batched=True, remove_columns=["user", "assistant"])
    val_ds = Dataset.from_list(val_rows).map(to_text, batched=True, remove_columns=["user", "assistant"])

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=BLOCK_SIZE,
            padding="max_length",
        )

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    return train_tok, val_tok
