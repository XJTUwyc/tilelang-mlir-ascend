from __future__ import annotations

from tvm import IRModule
from tvm.target import Target

import tilelang
from tilelang.backend.pass_pipeline import PassPipeline, register_pipeline


def TilePassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    del target
    return tilelang.transform.Simplify()(mod)


tile_pipeline = PassPipeline("tile", TilePassPipelineBody)
register_pipeline(tile_pipeline)
