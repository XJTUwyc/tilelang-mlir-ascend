"""Shared TileLangIR dialect configuration."""

from __future__ import annotations

from typing import Any


DIALECT_NAMESPACE = "tilelang"


def configure_context(context: Any) -> None:
    """Configure an MLIR context for the gradually introduced TileLangIR ops."""

    context.allow_unregistered_dialects = True
