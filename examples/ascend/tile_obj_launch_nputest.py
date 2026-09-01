""" E2E correctness test for a precompiled fused_matmul_bwd_w ``.o``.

Set ``OPENTILE_FUSED_MATMUL_BWD_W_OBJ`` to the object path, then run:

    python -m pytest examples/ascend/tile_obj_launch_nputest.py -x -s

``OPENTILE_TEST_UBUF`` defaults to zero. Set it to the dynamic UBUF bytes
reported by the compiler/manifest when the object requires dynamic UBUF.
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

def _npu_available():
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False

if not _npu_available():
    pytest.skip(
        "torch_npu is available, but no NPU device is visible",
        allow_module_level=True,
    )

_DEVICE_INDEX = int(os.environ.get("OPENTILE_TEST_DEVICE", "0").rsplit(":", 1)[-1])
torch.npu.set_device(_DEVICE_INDEX)

DTYPE = torch.float16
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
ATOL = 1e-3
RTOL = 1e-2
CASES = (
    pytest.param(512, 1024, 256, 0, id="source_m512_n1024_k256_grid4x8"),
    pytest.param(128, 1024, 128, 2202, id="compact_m128_n1024_k128_grid1x8"),
    pytest.param(1024, 128, 384, 2203, id="compact_m1024_n128_k384_grid8x1"),
)

def _object_path() -> Path:
    configured = os.environ.get("OPENTILE_FUSED_MATMUL_BWD_W_OBJ")
    if not configured:
        pytest.skip("set OPENTILE_FUSED_MATMUL_BWD_W_OBJ to fused_matmul_bwd_w_kernel.o")
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        pytest.fail(f"kernel object does not exist: {path}")
    return path

@pytest.mark.parametrize("m, n, k, seed", CASES)
def test_fused_matmul_bwd_w_obj(m, n, k, seed):
    from tilelang.opentile import load_tile_obj

    generator = torch.Generator(device="cpu").manual_seed(seed)
    x_cpu = torch.randn((k, m), dtype=DTYPE, generator=generator)
    dy_cpu = torch.randn((k, n), dtype=DTYPE, generator=generator)
    expected = (x_cpu.float().transpose(0, 1) @ dy_cpu.float()).to(DTYPE)

    device = torch.device("npu", torch.npu.current_device())
    x = x_cpu.to(device)
    dy = dy_cpu.to(device)
    dw = torch.full((m, n), float("nan"), dtype=DTYPE, device=device)
    lock_w = torch.zeros(32 * 1024, dtype=torch.int32, device=device)

    grid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = grid_m * grid_n
    ubuf_size = int(os.environ.get("OPENTILE_TEST_UBUF", "0"))
    kernel = load_tile_obj(
        _object_path(),
        kernel_name="fused_matmul_bwd_w_kernel",
        arg_types=["handle", "handle", "handle", "handle", "int32", "int32", "int32"],
        handle_dtypes=["float16", "float16", "float16", "int32"],
        grid=grid,
        ubuf_size=ubuf_size,
        binary_kind="aicore",
        grid_dims=(grid, 1, 1),
    )

    # Runtime ABI: dy_ptr, x_ptr, dw_ptr, LOCK_W, M, N, K, then the three 
    # trailing grid i32 (gridX/gridY/gridZ) injected automatically by 
    # load_tile_obj. The output is the third tensor.
    kernel(
        dy,
        x,
        dw,
        lock_w,
        m,
        n,
        k,
    )
    torch.npu.synchronize()

    actual = dw.cpu()
    finite = torch.isfinite(actual)
    assert bool(finite.all()), f"non-finite output: {actual.numel() - int(finite.sum().item())}/{actual.numel()}"
    torch.testing.assert_close(actual.float(), expected.float(), atol=ATOL, rtol=RTOL, equal_nan=False)

    diff = (actual.float() - expected.float()).abs()
    print(
        "[E2E_COMPARE] op=fused_matmul_bwd_w tensor=dw "
        f"shape=({m},{n}) k={k} grid={grid} logical_grid=({grid_m},{grid_n}) ubuf={ubuf_size} "
        f"pass=1 finite={actual.numel()}/{actual.numel()} "
        f"max_abs={float(diff.max().item()):.8g} "
        f"mean_abs={float(diff.mean().item()):.8g} "
        f"atol={ATOL} rtol={RTOL}",
        flush=True,
    )
