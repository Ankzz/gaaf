#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' "Error: uv is required but was not found on PATH." \
        "Install uv from https://docs.astral.sh/uv/getting-started/installation/ and rerun ./setup.sh." >&2
    exit 1
fi

printf '%s\n' "Installing Python 3.14 with uv..."
uv python install 3.14

printf '%s\n' "Creating the project environment and installing locked dependencies..."
uv sync --locked

printf '%s\n' "Setup complete. Run project commands with: uv run python -m gaaf.<module>"
