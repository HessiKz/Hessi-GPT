#!/usr/bin/env python3
# Hessi-GPT — https://github.com/HessiKz/
"""Fine-tune GPT-2 124M on chat-style dialog data."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from gpt2_chat.config import (
    BASE_MODEL,
    CHECKPOINT_DIR,
    EVAL_EVERY,
    GRAD_ACCUM,
    LEARNING_RATE,
    MAX_STEPS,
    SAVE_EVERY,
    TRAIN_BATCH_SIZE,
    WARMUP_STEPS,
)
from gpt2_chat.data import build_text_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 for chat")
    parser.add_argument("--output", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--resume", action="store_true", help="Resume from output checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    load_path = str(args.output) if args.resume and (args.output / "config.json").exists() else BASE_MODEL
    print(f"Loading model from {load_path}")

    tokenizer = AutoTokenizer.from_pretrained(load_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(load_path)
    model.to(device)

    print("Loading chat dataset…")
    train_ds, val_ds = build_text_dataset(tokenizer)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_steps=args.max_steps,
        warmup_steps=WARMUP_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_EVERY,
        save_steps=SAVE_EVERY,
        save_total_limit=2,
        logging_steps=20,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print(f"Fine-tuning for {args.max_steps} steps…")
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"Saved chat model to {args.output}")


if __name__ == "__main__":
    main()
