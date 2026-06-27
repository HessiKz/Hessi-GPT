# Hessi-GPT — https://github.com/HessiKz/
"""Chat prompt formatting for GPT-2 instruction-style fine-tuning."""

from __future__ import annotations

from gpt2_chat.config import ASSISTANT_TAG, TURN_END, USER_TAG


def format_turn(user: str, assistant: str) -> str:
    user = user.strip()
    assistant = assistant.strip()
    return f"{USER_TAG}: {user}\n{ASSISTANT_TAG}: {assistant}{TURN_END}"


def format_prompt(user: str, history: list[tuple[str, str]] | None = None) -> str:
    parts: list[str] = []
    if history:
        for past_user, past_assistant in history:
            parts.append(format_turn(past_user, past_assistant))
    parts.append(f"{USER_TAG}: {user.strip()}\n{ASSISTANT_TAG}:")
    return "".join(parts)


def extract_assistant_reply(generated: str, prompt: str) -> str:
    text = generated[len(prompt) :] if generated.startswith(prompt) else generated
    stop_markers = (f"\n{USER_TAG}:", f"\n{ASSISTANT_TAG}:", TURN_END)
    cut = len(text)
    for marker in stop_markers:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    reply = text[:cut].strip()
    return reply if reply else text.strip()
