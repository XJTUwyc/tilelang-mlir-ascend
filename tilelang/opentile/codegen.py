from __future__ import annotations

from tvm.target import Target

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen


def _is_tile_target(target: Target) -> bool:
    return "tile" in target.keys


register_device_codegen(
    "c",
    DeviceCodegen(
        "tile",
        build=global_func_device_codegen("target.build.tilelang_tile"),
        build_without_compile=global_func_device_codegen("target.build.tilelang_tile_without_compile"),
        supports_target=_is_tile_target,
    ),
    override=True,
)
