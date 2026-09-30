"""NPU integration tests for automatic Tile JIT compilation and launch.

Run from the repository root after configuring TILE_OPT, TILE_TRANSLATE and
CCEC.  DSA uses OPENTILE_DSA_DSL when its DSL is not next to this file.
No precompiled object or explicit TileLaunchSpec is supplied by these tests.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import tilelang
from tilelang.jit.adapter.tile import TileKernelAdapter
from tilelang.jit.kernel import JITKernel
from tilelang.opentile import ConstantBinding, ParameterBinding

from vadd_demo import vadd


BATCH, SEQ, SEQ_KV, HEADS, DIM, TOP_K = 1, 4096, 4096, 64, 256, 128
SPARSE_BLOCK = 64
DSA_DTYPES = ["bfloat16", "bfloat16", "float32", "int32", "float32"]
DSA_SHAPES = [
    (BATCH, SEQ, HEADS, DIM),
    (BATCH, SEQ_KV, DIM),
    (HEADS,),
    (BATCH, SEQ, TOP_K),
    (BATCH, SEQ, HEADS, DIM),
]


def _launch_record(kernel):
    prefix = "// tilelang.launch.v1 "
    first_line = kernel.artifact.kernel_source.splitlines()[0]
    assert first_line.startswith(prefix)
    return json.loads(first_line[len(prefix) :])


def _assert_common_jit(kernel):
    assert isinstance(kernel, JITKernel)
    assert kernel.execution_backend == "tile_obj"
    assert isinstance(kernel.adapter, TileKernelAdapter)
    assert kernel.torch_function is kernel.adapter.func
    assert kernel.adapter.compilation_result.object_bytes
    return kernel.adapter.compilation_result.launch_spec


@pytest.fixture(scope="module")
def npu_device():
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("no NPU device is visible")
    index = int(os.environ.get("OPENTILE_TEST_DEVICE", "0").rsplit(":", 1)[-1])
    torch.npu.set_device(index)
    return torch.device("npu", index)


@tilelang.jit(
    out_idx=[2],
    target="tile",
    execution_backend="auto",
    verbose=True,
)
def build_vadd_jit():
    return vadd()


@pytest.mark.parametrize("entry", ["compile", "jit"])
def test_vadd_jit(npu_device, entry):
    if entry == "compile":
        kernel = tilelang.compile(
            vadd(),
            out_idx=[2],
            target="tile",
            execution_backend="auto",
            verbose=True,
        )
    else:
        kernel = build_vadd_jit()

    spec = _assert_common_jit(kernel)
    assert kernel.out_idx == [2]
    assert spec.kernel_name == "main"
    assert spec.block_count == 32
    assert spec.dynamic_ubuf_bytes == 0
    assert [arg.kind for arg in spec.arguments[:3]] == ["handle"] * 3
    assert [arg.source for arg in spec.arguments[:3]] == [
        ParameterBinding(0),
        ParameterBinding(1),
        ParameterBinding(2),
    ]
    assert all(arg.kind == "int32" and arg.source == ConstantBinding(0) for arg in spec.arguments[3:])
    record = _launch_record(kernel)
    assert record == {
        "functions": [
            {
                "name": "main",
                "kinds": ["handle", "handle", "handle"],
                "axes": [{"tag": "blockIdx.x", "extent": 32}],
            }
        ]
    }

    generator = torch.Generator(device="cpu").manual_seed(20260910)
    a_cpu = torch.randn((1024, 256), dtype=torch.float32, generator=generator)
    b_cpu = torch.randn((1024, 256), dtype=torch.float32, generator=generator)
    output = kernel(a_cpu.to(npu_device), b_cpu.to(npu_device))
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), a_cpu + b_cpu, atol=1e-5, rtol=1e-5)
    print(
        f"[E2E_COMPARE] op=vadd entry={entry} backend={kernel.execution_backend} "
        f"wrapper=JITKernel pass=1 block_count={spec.block_count}",
        flush=True,
    )


def _load_dsa_program():
    configured = os.environ.get("OPENTILE_DSA_DSL")
    dsl_path = Path(configured).expanduser().resolve() if configured else Path(__file__).with_name("DSA_demo.py")
    if not dsl_path.is_file():
        pytest.skip(f"DSA DSL not found at {dsl_path}; set OPENTILE_DSA_DSL")
    module_spec = importlib.util.spec_from_file_location("tile_jit_test_dsa", dsl_path)
    if module_spec is None or module_spec.loader is None:
        pytest.fail(f"cannot import DSA DSL: {dsl_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    builder = getattr(module, "dsa_demo", None) or getattr(module, "flash_attention", None)
    if not callable(builder):
        pytest.fail(f"{dsl_path} must define dsa_demo() or flash_attention()")
    return builder(), dsl_path


@pytest.fixture(scope="module")
def dsa_jit_kernel(npu_device):
    program, dsl_path = _load_dsa_program()
    kernel = tilelang.compile(
        program,
        out_idx=[4],
        target="tile",
        execution_backend="auto",
        verbose=True,
    )
    spec = _assert_common_jit(kernel)
    assert kernel.out_idx == [4]
    assert kernel.artifact.params is not None
    assert len(kernel.artifact.params) == 5
    for param, shape, dtype in zip(kernel.artifact.params, DSA_SHAPES, DSA_DTYPES):
        assert tuple(int(dim) for dim in param.shape) == shape
        assert param.torch_dtype() == getattr(torch, dtype)

    record = _launch_record(kernel)
    assert record == {
        "functions": [
            {
                "name": "main",
                "kinds": ["handle"] * 5,
                "axes": [
                    {"tag": "blockIdx.x", "extent": SEQ},
                    {"tag": "blockIdx.y", "extent": 2},
                ],
            }
        ]
    }
    assert spec.kernel_name == "main"
    assert spec.block_count == SEQ
    assert spec.dynamic_ubuf_bytes == 0
    assert [arg.kind for arg in spec.arguments[:5]] == ["handle"] * 5
    assert [arg.source for arg in spec.arguments[:5]] == [ParameterBinding(i) for i in range(5)]
    assert all(arg.kind == "int32" and arg.source == ConstantBinding(0) for arg in spec.arguments[5:])
    abi = ",".join(arg.kind for arg in spec.arguments)
    print(
        f"[DSA_JIT_COMPILE] dsl={dsl_path} kernel={spec.kernel_name} "
        f"block_count={spec.block_count} backend={kernel.execution_backend} abi={abi}",
        flush=True,
    )
    return kernel, npu_device


def _dsa_reference(q, kv, sink, indices):
    q = q.float()
    kv = kv.float()
    rows, heads, dim = q.shape
    maximum = sink.float().expand(rows, heads).clone()
    denominator = torch.ones_like(maximum)
    output = torch.zeros((rows, heads, dim), dtype=torch.float32)
    for start in range(0, indices.shape[1], SPARSE_BLOCK):
        idx = indices[:, start : start + SPARSE_BLOCK].long()
        valid = idx != -1
        gathered = kv[idx.clamp_min(0)]
        gathered = gathered.masked_fill(~valid[..., None], 0.0)
        scores = torch.bmm(q, gathered.transpose(1, 2)) * (dim**-0.5)
        new_maximum = torch.maximum(maximum, scores.amax(dim=-1))
        probability = torch.exp(scores - new_maximum[..., None]) * valid[:, None, :]
        alpha = torch.exp(maximum - new_maximum)
        denominator = denominator * alpha + probability.sum(dim=-1)
        output = output * alpha[..., None] + torch.bmm(probability.to(torch.bfloat16).float(), gathered)
        maximum = new_maximum
    return output / denominator[..., None]


def _query_rows(count):
    if not 0 <= count <= SEQ:
        raise ValueError(f"OPENTILE_DSA_REF_QUERIES must be in [0, {SEQ}]")
    if count == 0:
        return torch.arange(SEQ)
    if count < 2:
        raise ValueError("use at least 2 reference queries to check both endpoints")
    return torch.linspace(0, SEQ - 1, count, dtype=torch.float64).round().long().unique()


@pytest.mark.parametrize("case", ["valid", "mixed_invalid", "all_invalid"])
def test_dsa_jit(dsa_jit_kernel, case):
    kernel, device = dsa_jit_kernel
    atol = float(os.environ.get("OPENTILE_DSA_ATOL", "0.003"))
    rtol = float(os.environ.get("OPENTILE_DSA_RTOL", "0.02"))
    if not (0 <= atol < float("inf") and 0 <= rtol < float("inf")):
        raise ValueError("comparison tolerances must be finite and nonnegative")
    selected = _query_rows(int(os.environ.get("OPENTILE_DSA_REF_QUERIES", "64")))
    generator = torch.Generator(device="cpu").manual_seed(20260907)
    q_cpu = torch.randn((BATCH, SEQ, HEADS, DIM), dtype=torch.bfloat16, generator=generator)
    kv_cpu = torch.randn((BATCH, SEQ_KV, DIM), dtype=torch.bfloat16, generator=generator)
    sink_cpu = torch.linspace(-1.0, 1.0, HEADS, dtype=torch.float32)
    starts = torch.randint(SEQ_KV, (BATCH, SEQ, 1), generator=generator)
    indices_cpu = ((starts + torch.arange(TOP_K) * 31) % SEQ_KV).to(torch.int32)
    if case == "mixed_invalid":
        invalid = torch.rand(indices_cpu.shape, generator=generator) < 0.25
        indices_cpu[invalid] = -1
        indices_cpu[:, 0, :] = -1
        indices_cpu[:, -1, :SPARSE_BLOCK] = -1
    elif case == "all_invalid":
        indices_cpu.fill_(-1)

    q, kv, sink, indices = (tensor.to(device) for tensor in (q_cpu, kv_cpu, sink_cpu, indices_cpu))
    output = kernel(q, kv, sink, indices)
    assert tuple(output.shape) == DSA_SHAPES[-1]
    assert output.dtype == torch.float32
    assert output.device == device
    torch.npu.synchronize()
    actual = output.cpu()[0]
    assert bool(torch.isfinite(actual).all())

    max_abs, error_sum, checked = 0.0, 0.0, 0
    if case == "all_invalid":
        torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
        checked = SEQ
    else:
        for rows in selected.split(8):
            expected = _dsa_reference(q_cpu[0, rows], kv_cpu[0], sink_cpu, indices_cpu[0, rows])
            got = actual[rows]
            torch.testing.assert_close(got, expected, atol=atol, rtol=rtol, equal_nan=False)
            difference = (got - expected).abs()
            max_abs = max(max_abs, float(difference.max()))
            error_sum += float(difference.double().sum())
            checked += rows.numel()
    print(
        f"[E2E_COMPARE] op=dsa entry=compile backend={kernel.execution_backend} "
        f"wrapper=JITKernel case={case} pass=1 "
        f"finite={actual.numel()}/{actual.numel()} checked_queries={checked}/{SEQ} "
        f"heads={HEADS} max_abs={max_abs:.8g} "
        f"mean_abs={error_sum / (checked * HEADS * DIM):.8g} "
        f"atol={atol} rtol={rtol}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-s"]))
