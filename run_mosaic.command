#!/usr/bin/env bash
# Mosaic launcher for macOS — double-clickable from Finder.
#
# First-time setup on this machine:
#   chmod +x run_mosaic.command
# (Finder runs .command files in Terminal but only if they're executable.)

set -e
cd "$(dirname "$0")"

# Keep the Terminal window open on any failure so the user can read messages.
trap 'echo ""; echo "Press Enter to close..."; read -r' EXIT

# ── Check Xcode Command Line Tools (needed to build geopandas / pyogrio deps) ──
if ! xcode-select -p &>/dev/null; then
    echo ""
    echo "Mosaic needs the macOS Command Line Tools to install its dependencies."
    echo ""
    echo "A system dialog should appear asking to install them."
    echo ""
    echo "Steps:"
    echo "  1) Click 'Install' in the dialog"
    echo "  2) Wait for the install to finish (10-15 minutes)"
    echo "  3) Double-click run_mosaic.command again"
    echo ""
    xcode-select --install 2>/dev/null || true
    exit 1
fi

# ── Locate or install uv ──────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "uv (Python environment manager) not found. Installing now..."
        echo "(one-time setup, requires internet)"
        echo ""
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        if ! command -v uv &>/dev/null; then
            echo ""
            echo "ERROR: uv was installed but is not reachable in this session."
            echo "Close this window, open a new Terminal, and double-click"
            echo "run_mosaic.command again."
            exit 1
        fi
    fi
fi

echo "Starting Mosaic..."
if [ ! -d ".venv" ]; then
    echo "First launch: installing dependencies (2-3 minutes). Subsequent launches are instant."
fi
echo ""

uv run python -m mosaic
ec=$?

if [ $ec -ne 0 ]; then
    echo ""
    echo "Mosaic exited with an error. See the messages above for details."
fi

# Disable the trap so a clean exit doesn't make the user press Enter.
if [ $ec -eq 0 ]; then
    trap - EXIT
fi
exit $ec
