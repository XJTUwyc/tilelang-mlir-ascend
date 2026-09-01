"""Launch-time argument dtype checks for tile object kernels.

What is verified
----------------
The launch-time checks added to ``tilelang/opentile/tile_obj.py``:

- tensor dtype must match the IR buffer dtype: same-width int/uint pairs
  pass (identical bit patterns), bfloat16 vs float16 is rejected (same
  width, different layout), unknown IR dtypes downgrade to a warning;
- scalar args are converted to the launcher's int64 representation: integer
  kinds use their integer value while float32/float64 use their IEEE-754 bit
  patterns; numpy scalar compatibility is covered when numpy is available;
- ``_encode_args`` reports a dtype mismatch before the device check
  (pairing correctness beats runtime usability).

Workflow
--------
1. Run all checks (torch/numpy-dependent ones SKIP when missing)::

       python examples/ascend/test_tile_obj_dtype.py

2. List the available checks::

       python examples/ascend/test_tile_obj_dtype.py --list

Shared helpers live in ``tile_obj_test_common.py``. Run from the repo root
with the project venv (``/home/wuyuchao/dev/.venv``).
"""

from __future__ import annotations

import struct
import sys
import warnings

from tile_obj_test_common import _expect_raises, run_module, stub_kernel, tile_obj


# ---------------------------------------------------------------------------
# Handle dtype pairing rules
# ---------------------------------------------------------------------------
# 测试点：handle dtype 不匹配必须报错（裸指针 ABI 下会静默算错）；
#         同宽 int/uint 互通（位模式相同）；bfloat 与 float16 必须区分；
#         未知 IR dtype 降级为警告；complex 拒绝。
# 验证功能：_check_handle_dtype 的配对正确性规则。
def check_handle_dtype_rules():
    """Handle dtype pairing: mismatch errors, same-width int/uint pass."""
    # mismatch rejected
    _expect_raises(TypeError, tile_obj._check_handle_dtype, 0, "float32", "float16")
    _expect_raises(TypeError, tile_obj._check_handle_dtype, 0, "float16", "float32")
    # bfloat16 vs float16: same width, different bit layout -> error
    _expect_raises(TypeError, tile_obj._check_handle_dtype, 0, "bfloat16", "float16")
    # torch bfloat16 maps to TIR "bfloat"
    tile_obj._check_handle_dtype(0, "bfloat16", "bfloat")
    # same-width int/uint share the bit pattern -> pass
    tile_obj._check_handle_dtype(0, "int16", "uint16")
    tile_obj._check_handle_dtype(0, "uint8", "int8")
    # different width / family -> error
    _expect_raises(TypeError, tile_obj._check_handle_dtype, 0, "int32", "int16")
    _expect_raises(TypeError, tile_obj._check_handle_dtype, 0, "complex64", "float32")
    # unknown expected dtype -> warn, not raise
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tile_obj._check_handle_dtype(0, "float8_e4m3fn", "fp8e4m3")
    assert caught, "unknown IR dtype should warn"
    assert "fp8e4m3" in str(caught[0].message)


# ---------------------------------------------------------------------------
# Scalar argument ABI encoding
# ---------------------------------------------------------------------------
# 测试点：整数参数转换为 int；float32/float64 参数转换为对应的
#         IEEE-754 位模式。类型范围和溢出检查由热路径中移除。
# 验证功能：_encode_scalar_arg 的 ABI 编码规则。
def check_scalar_arg_rules():
    """Scalar args use the representation expected by the C++ launcher."""
    assert tile_obj._encode_scalar_arg("int8", 127) == 127
    assert tile_obj._encode_scalar_arg("int32", -3) == -3
    assert tile_obj._encode_scalar_arg("uint64", 5) == 5
    assert tile_obj._encode_scalar_arg("int32", True) == 1

    float32_value = 1.5
    assert tile_obj._encode_scalar_arg("float32", float32_value) == struct.unpack(
        "<i", struct.pack("<f", float32_value)
    )[0]

    float64_value = -2.25
    assert tile_obj._encode_scalar_arg("float64", float64_value) == struct.unpack(
        "<q", struct.pack("<d", float64_value)
    )[0]


# 测试点：精简编码路径仍兼容 numpy 标量；无 numpy 时 SKIP。
# 验证功能：_encode_scalar_arg 接受支持 Python 数值协议的标量。
def check_scalar_numpy_compatibility():
    """numpy scalars are accepted without an explicit .item() call."""
    try:
        import numpy as np
    except ImportError:
        print("SKIP: numpy not installed")
        return
    assert tile_obj._encode_scalar_arg("int64", np.int64(5)) == 5
    assert tile_obj._encode_scalar_arg("float32", np.float32(1.5)) == struct.unpack(
        "<i", struct.pack("<f", 1.5)
    )[0]


# ---------------------------------------------------------------------------
# _encode_args integration (torch)
# ---------------------------------------------------------------------------
# 测试点：torch tensor 的 dtype/device 校验顺序——dtype 错误先于 device
#         错误报告（配对正确性优先于运行时可用性）；无 torch 时 SKIP。
# 验证功能：TileObjKernel._encode_args 的 handle 分支。
def check_encode_args_torch():
    """_encode_args: dtype mismatch reported before device check."""
    try:
        torch = tile_obj._torch_module()
    except ImportError:
        print("SKIP: torch not installed")
        return
    info = tile_obj.LaunchInfo(
        name="main",
        arg_types=["handle", "float32"],
        handle_shapes=[],
        handle_dtypes=["float16"],
        grid_exprs=[],
        trailing_grid_dims=None,
    )
    kernel = stub_kernel(info)
    wrong_dtype = torch.zeros(4, dtype=torch.float32)  # CPU on purpose: dtype must fire first
    _expect_raises(TypeError, kernel._encode_args, [wrong_dtype, 1.0])
    right_dtype = torch.zeros(4, dtype=torch.float16)
    _expect_raises(ValueError, kernel._encode_args, [right_dtype, 1.0])  # CPU -> device error


if __name__ == "__main__":
    sys.exit(run_module(globals()))
