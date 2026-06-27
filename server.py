# Hessi-GPT — https://github.com/HessiKz/
"""FastAPI server for Hessi-GPT chat UI with SSE pipeline telemetry."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import (
    DEFAULT_CHECKPOINT,
    EngineState,
    PIPELINE_STEPS,
    generate_stream,
    get_architecture_summary,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
engine_state = EngineState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    for _ in generate_stream(engine_state, "Hello", max_tokens=8, record_history=False):
        pass
    yield


app = FastAPI(title="Hessi-GPT", version="0.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    max_tokens: int = Field(default=120, ge=1, le=400)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    ready = engine_state.model is not None and engine_state.tokenizer is not None
    return {
        "status": "ready" if ready else "booting",
        "model_loaded": ready,
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "backend": "pytorch-gpt2",
        "author": "HessiKz",
        "github": "https://github.com/HessiKz/",
    }


@app.get("/api/architecture")
async def architecture() -> dict:
    return {
        "steps": [{"id": s[0], "label": s[1]} for s in PIPELINE_STEPS],
        "spec": get_architecture_summary(),
        "blocks": [
            "Token + Position Embedding",
            "Masked Multi-Head Self-Attention",
            "LayerNorm + GELU FFN",
            "LM Head (tied embeddings)",
        ],
    }


async def sse_events(request: Request, prompt: str, max_tokens: int) -> AsyncIterator[str]:
    for event in generate_stream(engine_state, prompt, max_tokens=max_tokens):
        if await request.is_disconnected():
            break
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    return StreamingResponse(
        sse_events(request, body.prompt.strip(), body.max_tokens),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
