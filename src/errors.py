"""Error type used to report command failures with a clear message."""

from __future__ import annotations


class RagError(Exception):
    """Raised when a command cannot complete for a user-visible reason.

    The command-line entry point catches this error, prints its message
    without a traceback, and exits with a non-zero status.
    """
