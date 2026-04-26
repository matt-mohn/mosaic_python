#!/usr/bin/env bash
set -e

# Move to the directory containing this script
cd "$(dirname "$0")"

# ── Locate or install uv ──────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "uv not found. Installing now (one-time setup, requires internet)..."
        echo ""
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        if ! command -v uv &>/dev/null; then
            echo ""
            echo "ERROR: uv was installed but is not reachable in this session."
            echo "Close this terminal, open a new one, and run: ./run_mosaic.sh"
            exit 1
        fi
    fi
fi

echo "Starting MosaicPy Demo..."
echo "(First launch downloads and installs dependencies -- this takes 2-3 minutes.)"
echo "Subsequent launches are instant."
echo ""

uv run python -m mosaic.gui.app
