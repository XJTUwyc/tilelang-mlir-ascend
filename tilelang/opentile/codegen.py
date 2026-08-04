from __future__ import annotations

from tvm.target import Target

from tilelang.backend.device_codegen import DeviceCodegen, register_device_codegen
from tilelangir.codegen import build_tilelang_ir, build_tilelang_ir_without_compile


def _is_tile_target(target: Target) -> bool:
    return "tile" in target.keys


register_device_codegen(
    "c",
    DeviceCodegen(
        "tile",
        build=build_tilelang_ir,
        build_without_compile=build_tilelang_ir_without_compile,
        supports_target=_is_tile_target,
    ),
    override=True,
)
