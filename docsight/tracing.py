"""Optional Langfuse tracing.

If LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set and the langfuse
package is installed, each question becomes a trace with child spans for
retrieval, generation and verification. Otherwise every call is a no-op, so
the app never depends on the tracing backend being up.
"""

from __future__ import annotations

import os
from contextlib import contextmanager


class _NoopSpan:
    def update(self, **_):
        pass


_client = None
_enabled = None


def _get_client():
    global _client, _enabled
    if _enabled is None:
        _enabled = False
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import get_client  # langfuse >= 3

                _client = get_client()
                _enabled = True
            except Exception:
                _enabled = False
    return _client if _enabled else None


@contextmanager
def span(name: str, **attrs):
    client = _get_client()
    if client is None:
        yield _NoopSpan()
        return
    try:
        cm = client.start_as_current_span(name=name, input=attrs.get("input"), metadata=attrs.get("metadata"))
    except Exception:
        yield _NoopSpan()
        return
    with cm as s:
        yield s


def flush() -> None:
    client = _get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
