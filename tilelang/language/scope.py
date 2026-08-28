"""Execution-scope constructs exposed by the TileLang language surface."""

from __future__ import annotations

from typing import Any

from tvm.tirx.script.builder.ir import sblock, sblock_attr


_EXECUTION_SCOPE_ATTR = "tl.execution_scope"
_SIMT_THREADS_ATTR = "tl.simt_threads"
_DEFAULT_SIMT_THREADS = 1024
_MAX_UINT32 = (1 << 32) - 1


class _ExecutionScope:
    """Attach execution semantics to an otherwise ordinary Core TIRX block."""

    def __init__(self, name: str, annotations: dict[str, Any]):
        self._frame = sblock(name)
        self._annotations = annotations

    def __enter__(self):
        result = self._frame.__enter__()
        sblock_attr(self._annotations)
        return result

    def __exit__(self, exc_type, exc_value, traceback):
        return self._frame.__exit__(exc_type, exc_value, traceback)


def _normalize_simt_threads(threads: int) -> int:
    if isinstance(threads, bool) or not isinstance(threads, int):
        raise TypeError(
            f"SimtVF threads must be a uint32 integer, got {type(threads).__name__}"
        )
    if not 1 <= threads <= _MAX_UINT32:
        raise ValueError("SimtVF threads must be in the uint32 range [1, 4294967295]")
    return threads


def SimdVF() -> _ExecutionScope:  # pylint: disable=invalid-name
    """Create a SIMD vector-function scope using an annotated Core TIRX block."""
    return _ExecutionScope("SimdVF", {_EXECUTION_SCOPE_ATTR: "simd"})


def SimtVF(threads: int = _DEFAULT_SIMT_THREADS) -> _ExecutionScope:  # pylint: disable=invalid-name
    """Create a SIMT vector-function scope using common Core TIRX constructs.

    ``threads`` is the requested physical worker count, represented as a
    positive uint32 and defaulting to 1024. Logical work remains expressed by
    ordinary :func:`T.Parallel` loops inside the scope, allowing a backend pass
    to map logical work onto physical workers later.
    """
    return _ExecutionScope(
        "SimtVF",
        {
            _EXECUTION_SCOPE_ATTR: "simt",
            _SIMT_THREADS_ATTR: _normalize_simt_threads(threads),
        },
    )
