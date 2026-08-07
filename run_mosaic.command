#!/usr/bin/env bash
# Mosaic launcher for macOS -- double-clickable from Finder.
#
# Mode 100755 is tracked in git and survives GitHub's zip. If Finder still
# refuses to run this: chmod +x run_mosaic.command

set -e

# Set before the first cd, so a failing cd still leaves the window open.
trap 'echo ""; echo "Press Enter to close..."; read -r' EXIT

cd "$(dirname "$0")"

# Opening this straight out of the archive leaves nothing to run.
if [ ! -f "pyproject.toml" ]; then
    echo ""
    echo "ERROR: Mosaic's files are missing from this folder."
    echo ""
    echo "Extract the whole .zip to a real folder first, then run"
    echo "run_mosaic.command from the extracted copy."
    exit 1
fi

# ---- Apple Silicon check -----------------------------------------------------
# Dear PyGui publishes macOS wheels only for arm64, with no source fallback,
# so dependency resolution fails outright on an Intel Mac.
if [ "$(uname -m)" != "arm64" ]; then
    echo ""
    echo "Mosaic cannot run on an Intel Mac."
    echo ""
    echo "Its GUI toolkit ships macOS builds only for Apple Silicon (M1 and"
    echo "later), and there is no version that can be built from source."
    echo "Mosaic also requires macOS 13 or newer."
    exit 1
fi

# ---- Keep the environment out of synced folders ------------------------------
# The venv is thousands of files; outside the Mosaic folder it never syncs.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.local/share/mosaic/venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.local/share/mosaic/uv-cache}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# ---- Check Xcode Command Line Tools ------------------------------------------
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

# ---- Locate or install uv ----------------------------------------------------
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
            echo "ERROR: uv could not be downloaded or is not reachable."
            echo "If you are on a managed Mac, a firewall or proxy may be"
            echo "blocking it. Download uv manually from"
            echo "  https://github.com/astral-sh/uv/releases"
            echo "place it at $HOME/.local/bin/uv, then try again."
            exit 1
        fi
    fi
fi

echo "Starting Mosaic..."
if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
    echo "First launch: installing dependencies (2-3 minutes, about 500 MB)."
    echo "Subsequent launches are instant."
fi
echo ""

# "|| ec=$?" stops set -e from aborting before the error message below.
ec=0
uv run python -m mosaic || ec=$?

if [ $ec -ne 0 ]; then
    echo ""
    echo "Mosaic exited with an error. The details are in the messages above."
fi

# Disable the trap so a clean exit doesn't make the user press Enter.
if [ $ec -eq 0 ]; then
    trap - EXIT
fi
exit $ec
