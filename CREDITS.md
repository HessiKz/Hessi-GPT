# Credits

## Project

| | |
|---|---|
| **Name** | Hessi-GPT (GPT-2.5) |
| **Author** | [HessiKz](https://github.com/HessiKz/) |
| **GitHub** | [https://github.com/HessiKz/Hessi-GPT](https://github.com/HessiKz/Hessi-GPT) |
| **Demo** | [https://hessikz.github.io/Hessi-GPT/](https://hessikz.github.io/Hessi-GPT/) |
| **License** | MIT — see [LICENSE](LICENSE) |

## Author contributions

- **HESSI-GPT (GPT-2.5)** — decoder-only chat language model written and chat-fine-tuned in this repo
- Chat fine-tuning pipeline (`finetune_gpt2_chat.py`, `gpt2_chat/`)
- PyTorch streaming inference engine (`engine.py`)
- FastAPI SSE chat server (`server.py`)
- Industrial-brutalist chat UI (`web/`)
- CLI inference tool (`generate.py`)
- Project documentation, branding, and deployment scripts

## Models

| Model | Source | Use |
|-------|--------|-----|
| **HESSI-GPT (GPT-2.5) chat** | Written and chat-tuned in this project | Primary chat inference |
| **MiniGPT research path** | [jongoiko/minigpt](https://github.com/jongoiko/minigpt) | Original JAX training codebase lineage |

## Upstream

This project extends [jongoiko/minigpt](https://github.com/jongoiko/minigpt) for the early JAX/Equinox research path (small decoder-only Transformer with RoPE, RMSNorm, and SwiGLU on SimpleStories). The production chat product is **HESSI-GPT (GPT-2.5)** as documented in the README.

---

**© 2025–2026 [HessiKz](https://github.com/HessiKz/)**
