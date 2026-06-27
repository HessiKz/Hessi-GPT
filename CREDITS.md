# Credits

## Project

| | |
|---|---|
| **Name** | Hessi-GPT |
| **Author** | [HessiKz](https://github.com/HessiKz/) |
| **GitHub** | [https://github.com/HessiKz/Hessi-GPT](https://github.com/HessiKz/Hessi-GPT) |
| **License** | MIT — see [LICENSE](LICENSE) |

## Author contributions

- GPT-2 124M chat fine-tuning pipeline (`finetune_gpt2_chat.py`, `gpt2_chat/`)
- PyTorch streaming inference engine (`engine.py`)
- FastAPI SSE chat server (`server.py`)
- Industrial-brutalist chat UI (`web/`)
- CLI inference tool (`generate.py`)
- Project documentation, branding, and deployment scripts

## Models

| Model | Source | Use |
|-------|--------|-----|
| **GPT-2 124M (chat)** | [gpt2](https://huggingface.co/gpt2) + local fine-tune | Primary chat inference |
| **MiniGPT 29M** | [jongoiko/minigpt](https://github.com/jongoiko/minigpt) | Original JAX training codebase |

## Upstream

This project extends [jongoiko/minigpt](https://github.com/jongoiko/minigpt) — a small decoder-only Transformer with RoPE, RMSNorm, and SwiGLU, trained on SimpleStories.

---

**© 2025 [HessiKz](https://github.com/HessiKz/)**
