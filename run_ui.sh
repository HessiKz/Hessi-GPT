#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "Starting Hessi-GPT chat UI at http://127.0.0.1:8765"
echo "Loads GPT-2 124M chat checkpoint from checkpoints/gpt2-chat/"
exec .venv/bin/uvicorn server:app --host 127.0.0.1 --port 8765 --reload
