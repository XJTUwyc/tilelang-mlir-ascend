"""Minimal TIRX to TileLangIR MLIR translator."""

from __future__ import annotations

from tvm import IRModule

from .dialect import configure_context


class TileLangIRTranslator:
    """Translate a TIRX module into a TileLangIR MLIR module.

    The initial implementation deliberately emits only the top-level MLIR
    module. TIRX operation conversion is added incrementally by later changes.
    """

    def translate(self, source_module: IRModule) -> str:
        if not isinstance(source_module, IRModule):
            raise TypeError(
                "TileLangIR codegen expects tvm.IRModule, "
                f"but received {type(source_module).__name__}."
            )

        try:
            from mlir import ir
        except ImportError as exc:
            raise ImportError(
                "TileLangIR codegen requires LLVM's official MLIR Python bindings. "
                "Install them with `python -m pip install mlir-python-bindings "
                "-f https://llvm.github.io/eudsl`."
            ) from exc

        with ir.Context() as context:
            configure_context(context)
            with ir.Location.unknown(context):
                module = ir.Module.create()
                return str(module)
