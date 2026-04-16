#!/usr/bin/env bash
# BonsaiChat — One-click setup (macOS / Linux)
# agent-zero-features branch

set -e
echo
echo " ========================================"
echo "  BonsaiChat Setup"
echo " ========================================"
echo

# ── 1. Python check ──────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.10+ first."
    exit 1
fi
echo "[OK] $(python3 --version)"

# ── 2. venv ──────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "[*] Creating virtual environment…"
    python3 -m venv .venv
fi
echo "[OK] Virtual environment ready."

# ── 3. Install deps ──────────────────────────────────────────────────────
echo "[*] Installing Python dependencies (may take a few minutes)…"
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt --quiet
echo "[OK] Dependencies installed."

# ── 4. Directories ───────────────────────────────────────────────────────
mkdir -p bin models memory_data
echo "[OK] Directories ready."

# ── 5. llama-server check ────────────────────────────────────────────────
if [ ! -f "bin/llama-server" ]; then
    echo
    echo "[NOTICE] bin/llama-server not found."
    echo "  Download from: https://github.com/ggml-org/llama.cpp/releases"
    echo "  Place the llama-server binary inside the bin/ folder,"
    echo "  then run:  chmod +x bin/llama-server"
    echo
else
    chmod +x bin/llama-server
    echo "[OK] llama-server found."
fi

# ── 6. .env ──────────────────────────────────────────────────────────────
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "[OK] .env created from template."
fi

echo
echo " ========================================"
echo "  Setup complete!"
echo
echo "  To run BonsaiChat:"
echo "    .venv/bin/python app.py"
echo
echo "  To build the executable (Linux/macOS):"
echo "    .venv/bin/pyinstaller BonsaiChat.spec"
echo " ========================================"
echo
