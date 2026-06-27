"""Generate chat replies from Hessi-GPT (GPT-2 124M chat).

Author: HessiKz — https://github.com/HessiKz/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from engine import DEFAULT_CHECKPOINT, EngineState, generate_stream


def generate(prompt: str, checkpoint: Path, max_tokens: int) -> str:
    state = EngineState(checkpoint_path=checkpoint)
    result = prompt
    for event in generate_stream(state, prompt, max_tokens=max_tokens):
        if event["type"] == "done":
            result = event["text"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Hessi-GPT")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", default="Hello! How are you?")
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint} …", flush=True)
    print(generate(args.prompt, args.checkpoint, args.max_tokens), flush=True)


if __name__ == "__main__":
    main()
