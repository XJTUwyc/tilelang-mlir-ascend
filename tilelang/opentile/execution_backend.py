"""Execution backend for Tile JIT compiler-owned object bytes."""
# Keep generic CPU registrations; the resolver prefers this Tile-specific match.

from tilelang.cpu import execution_backend as _cpu_execution_backend
from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend
from .target import target_is_tile

register_execution_backend(
    "c",
    ExecutionBackendSpec(
        "tile_obj",
        supports_target=target_is_tile,
        enable_host_codegen=False,
        enable_device_compile=False,
    ),
)
