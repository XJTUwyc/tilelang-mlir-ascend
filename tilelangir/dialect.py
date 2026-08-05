"""Shared TileLangIR dialect configuration."""

from __future__ import annotations

from typing import Any

from ._tilelang_ops_gen import _Dialect

DIALECT_NAMESPACE = "tilelang"


def configure_context(context: Any) -> None:
    """Configure an MLIR context for TileLangIR dialect and ops.

    The custom ``tilelang`` dialect is registered at the Python level via
    ``@register_dialect`` / ``@register_operation`` decorators.  Because the
    dialect is not a C++-registered dialect, we keep
    ``allow_unregistered_dialects`` enabled so that the MLIR parser can
    round-trip ``tilelang.*`` operations.
    """

    context.allow_unregistered_dialects = True
