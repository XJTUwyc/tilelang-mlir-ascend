"""TileLang TIRX to TileLangIR MLIR code generation."""

# Import _tilelang_ops_gen first to trigger @register_dialect / @register_operation
# side-effects before any MLIR context is created.
from . import _tilelang_ops_gen  # noqa: F401

from ._tilelang_ops_gen import CopyOp, GemmOp, ScopeOp  # noqa: F401
from .codegen import build_tilelang_ir, build_tilelang_ir_without_compile
from .dialect import DIALECT_NAMESPACE, configure_context
from .translator import TileLangIRTranslator

__all__ = [
    "TileLangIRTranslator",
    "build_tilelang_ir",
    "build_tilelang_ir_without_compile",
    "CopyOp",
    "GemmOp",
    "ScopeOp",
    "DIALECT_NAMESPACE",
    "configure_context",
]
