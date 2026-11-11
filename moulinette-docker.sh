#!/usr/bin/env bash
# Run the Linux x86-64 moulinette binary through Docker on macOS.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MOULINETTE_BINARY="${MOULINETTE_BINARY:-$PROJECT_DIR/moulinette-ubuntu}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed." >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "Error: Start Docker Desktop before running the moulinette." >&2
    exit 1
fi

if [ ! -f "$MOULINETTE_BINARY" ]; then
    echo "Error: Moulinette binary not found: $MOULINETTE_BINARY" >&2
    echo "Place moulinette-ubuntu at the project root." >&2
    exit 1
fi

MOULINETTE_BINARY="$(
    cd "$(dirname "$MOULINETTE_BINARY")" && pwd
)/$(basename "$MOULINETTE_BINARY")"

exec docker run --rm --platform linux/amd64 \
    --mount "type=bind,src=$PROJECT_DIR,dst=$PROJECT_DIR,readonly" \
    --mount "type=bind,src=$MOULINETTE_BINARY,dst=/moulinette,readonly" \
    --workdir "$PROJECT_DIR" \
    ubuntu:22.04 /moulinette "$@"
