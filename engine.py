# Hessi-GPT — https://github.com/HessiKz/
"""GPT-2 124M chat inference engine with streaming pipeline events."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

from gpt2_chat.config import CHECKPOINT_DIR, GENERATION_KWARGS
from gpt2_chat.formatting import extract_assistant_reply, format_prompt

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = CHECKPOINT_DIR

PIPELINE_STEPS = [
    ("REQUEST_ACCEPT", "Accept user message"),
    ("LOAD_TOKENIZER", "Load GPT-2 BPE tokenizer"),
    ("RESTORE_CHECKPOINT", "Load GPT-2 124M chat weights"),
    ("ENCODE_PROMPT", "Format and encode chat prompt"),
    ("AUTOREGRESSIVE_DECODE", "Sample autoregressive tokens"),
    ("EMIT_RESPONSE", "Finalize assistant message"),
]


@dataclass
class EngineState:
    model: AutoModelForCausalLM | None = None
    tokenizer: AutoTokenizer | None = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    history: list[tuple[str, str]] = field(default_factory=list)


def _event(
    event_type: str,
    *,
    phase: str | None = None,
    status: str | None = None,
    detail: str | None = None,
    elapsed_ms: float | None = None,
    **payload: Any,
) -> dict[str, Any]:
    data: dict[str, Any] = {"type": event_type}
    if phase is not None:
        data["phase"] = phase
    if status is not None:
        data["status"] = status
    if detail is not None:
        data["detail"] = detail
    if elapsed_ms is not None:
        data["elapsed_ms"] = round(elapsed_ms, 1)
    data.update(payload)
    return data


def get_architecture_summary() -> list[dict[str, str]]:
    return [
        {"label": "PARAM COUNT", "value": "~124M"},
        {"label": "LAYERS", "value": "12"},
        {"label": "D_MODEL", "value": "768"},
        {"label": "HEADS", "value": "12"},
        {"label": "FFN DIM", "value": "3072"},
        {"label": "CONTEXT", "value": "1024"},
        {"label": "VOCAB", "value": "50257"},
        {"label": "POSITION", "value": "Learned"},
        {"label": "NORM", "value": "LayerNorm"},
        {"label": "FFN TYPE", "value": "GELU MLP"},
        {"label": "DECODE", "value": "Top-p sample"},
    ]


def _resolve_checkpoint(path: Path) -> Path:
    if (path / "config.json").exists():
        return path
    raise FileNotFoundError(
        f"No GPT-2 chat checkpoint at {path}. Run: python finetune_gpt2_chat.py"
    )


def ensure_engine(state: EngineState) -> Generator[dict[str, Any], None, None]:
    if state.model is not None and state.tokenizer is not None:
        return

    checkpoint = _resolve_checkpoint(state.checkpoint_path)
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()
    yield _event("step", phase="LOAD_TOKENIZER", status="running", detail=str(checkpoint))
    state.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if state.tokenizer.pad_token is None:
        state.tokenizer.pad_token = state.tokenizer.eos_token
    yield _event(
        "step",
        phase="LOAD_TOKENIZER",
        status="complete",
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        vocab_size=state.tokenizer.vocab_size,
    )

    t0 = time.perf_counter()
    yield _event("step", phase="RESTORE_CHECKPOINT", status="running", detail=str(checkpoint))
    state.model = AutoModelForCausalLM.from_pretrained(checkpoint)
    state.model.to(state.device)
    state.model.eval()
    param_count = sum(p.numel() for p in state.model.parameters())
    yield _event(
        "step",
        phase="RESTORE_CHECKPOINT",
        status="complete",
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        params=param_count,
    )


def generate_stream(
    state: EngineState,
    prompt: str,
    max_tokens: int = 120,
    *,
    record_history: bool = True,
) -> Iterator[dict[str, Any]]:
    yield _event("step", phase="REQUEST_ACCEPT", status="complete", detail=prompt[:120])

    for boot_event in ensure_engine(state):
        yield boot_event

    assert state.model is not None and state.tokenizer is not None

    chat_prompt = format_prompt(prompt, state.history)
    t0 = time.perf_counter()
    yield _event("step", phase="ENCODE_PROMPT", status="running")
    inputs = state.tokenizer(chat_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(state.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(state.device)
    prompt_len = input_ids.shape[1]
    yield _event(
        "step",
        phase="ENCODE_PROMPT",
        status="complete",
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        prompt_tokens=prompt_len,
        token_ids=input_ids[0, :32].tolist(),
    )

    gen_kwargs = {
        **GENERATION_KWARGS,
        "max_new_tokens": max_tokens,
        "pad_token_id": state.tokenizer.eos_token_id,
    }

    streamer = TextIteratorStreamer(
        state.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    yield _event("step", phase="AUTOREGRESSIVE_DECODE", status="running")

    generation_thread = Thread(
        target=state.model.generate,
        kwargs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "streamer": streamer,
            **gen_kwargs,
        },
    )
    generation_thread.start()

    partial = ""
    tokens_generated = 0
    stop_markers = ("\nUser:", "\nAssistant:")

    for piece in streamer:
        t_token = time.perf_counter()
        partial += piece
        tokens_generated += 1

        if any(marker in partial for marker in stop_markers):
            for marker in stop_markers:
                idx = partial.find(marker)
                if idx != -1:
                    partial = partial[:idx].rstrip()
            break

        yield _event(
            "token",
            phase="AUTOREGRESSIVE_DECODE",
            status="streaming",
            index=tokens_generated,
            token_id=-1,
            token_text=piece,
            partial=partial,
            context_len=prompt_len + tokens_generated,
            latency_ms=(time.perf_counter() - t_token) * 1000,
        )

    generation_thread.join()

    reply = extract_assistant_reply(partial, "")
    if not reply:
        reply = partial.strip()

    state.history.append((prompt, reply))
    if len(state.history) > 6:
        state.history = state.history[-6:]

    if not record_history:
        state.history.pop()

    yield _event(
        "step",
        phase="AUTOREGRESSIVE_DECODE",
        status="complete",
        tokens_generated=tokens_generated,
    )

    yield _event("step", phase="EMIT_RESPONSE", status="running")
    yield _event(
        "done",
        phase="EMIT_RESPONSE",
        status="complete",
        text=reply,
        tokens_generated=tokens_generated,
        prompt_tokens=prompt_len,
    )
