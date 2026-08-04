"""Python device-codegen entry points for TileLangIR."""

from __future__ import annotations

from tvm import IRModule
from tvm.target import Target

from tilelang import tvm

from .translator import TileLangIRTranslator


def _build_source_module(mod: IRModule) -> tvm.ffi.Module:
    source = TileLangIRTranslator().translate(mod)
    create_source_module = tvm.ffi.get_global_func("runtime.CSourceModuleCreate")
    return create_source_module(source, "mlir", [], [])


def build_tilelang_ir(mod: IRModule, target: Target) -> tvm.ffi.Module:
    """Generate and package TileLangIR MLIR without backend optimization."""

    del target
    return _build_source_module(mod)


def build_tilelang_ir_without_compile(mod: IRModule, target: Target) -> tvm.ffi.Module:
    """Generate TileLangIR MLIR without invoking an external compiler."""

    del target
    return _build_source_module(mod)
