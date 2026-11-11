"""RAG command-line entry point."""

from __future__ import annotations

import sys

import fire

from src.cli import RagCli
from src.errors import RagError


def main() -> int:
    """Run the Python Fire interface and report failures clearly.

    Returns:
        The process exit status: 0 on success, non-zero on failure.
    """
    try:
        fire.Fire(RagCli())
    except RagError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Error: interrupted by the user.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: unexpected failure: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
