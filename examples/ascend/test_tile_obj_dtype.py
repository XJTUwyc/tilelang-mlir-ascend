"""Launch-time argument dtype checks for TODO 2 (parameter dtype validation).

What is verified
----------------
The launch-time checks added to ``tilelang/opentile/tile_obj.py``:

- tensor dtype must match the IR buffer dtype: same-width int/uint pairs
  pass (identical bit patterns), bfloat16 vs float16 is rejected (same
  width, different layout), unknown IR dtypes downgrade to a warning;
- scalar args: floats/strings/None rejected for int kinds, per-kind value
  ranges enforced (the C++ packer truncates silently otherwise), float32
  overflow detected, numpy/torch scalar wrappers normalized via ``.item()``;
- ``_encode_args`` reports a dtype mismatch before the device check
  (pairing correctness beats runtime usability).

Workflow
--------
1. Run all checks (torch/numpy-dependent ones SKIP when missing)::

       python examples/ascend/test_tile_obj_dtype.py

2. List the available checks::

       python examples/ascend/test_tile_obj_dtype.py --list

Shared helpers live in ``tile_obj_test_common.py``; the TODO 3 bind-time
contract checks live in ``test_tile_obj_contract.py``.  Run from the repo
root with the project venv (``/home/wuyuchao/dev/.venv``).
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
# Scalar arg type / range / overflow rules
# ---------------------------------------------------------------------------
# 测试点：标量参数类型/范围校验：float 传 int kind 拒绝、超范围拒绝、
#         numpy/torch 标量包装器归一化、float32 溢出拒绝、边界值放行。
# 验证功能：_encode_scalar_arg 的类型与范围规则。
def check_scalar_arg_rules():
    """Scalar args: type/range checks, wrapper normalization, overflow."""
    # float passed to an int kind: previously truncated silently by int()
    _expect_raises(TypeError, tile_obj._encode_scalar_arg, 0, "int32", 1.5)
    _expect_raises(TypeError, tile_obj._encode_scalar_arg, 0, "int32", 1.0)
    # str / None rejected
    _expect_raises(TypeError, tile_obj._encode_scalar_arg, 0, "int32", "5")
    _expect_raises(TypeError, tile_obj._encode_scalar_arg, 0, "float32", None)
    # out-of-range values: previously truncated silently by the C++ packer
    _expect_raises(ValueError, tile_obj._encode_scalar_arg, 0, "int8", 300)
    _expect_raises(ValueError, tile_obj._encode_scalar_arg, 0, "uint8", -1)
    _expect_raises(ValueError, tile_obj._encode_scalar_arg, 0, "uint64", -(2**64))
    # boundary values pass and encode unchanged
    assert tile_obj._encode_scalar_arg(0, "int8", 127) == 127
    assert tile_obj._encode_scalar_arg(0, "int8", -128) == -128
    assert tile_obj._encode_scalar_arg(0, "uint8", 255) == 255
    # bool is an int subclass with unambiguous semantics
    assert tile_obj._encode_scalar_arg(0, "int32", True) == 1
    # float32 overflow: previously packed to inf silently
    _expect_raises(OverflowError, tile_obj._encode_scalar_arg, 0, "float32", 1e300)
    # float64 accepts the same value fine
    tile_obj._encode_scalar_arg(0, "float64", 1e300)
    # int passed to a float kind is a lossless widening
    assert tile_obj._encode_scalar_arg(0, "float32", 2) == struct.unpack("<i", struct.pack("<f", 2.0))[0]


# 测试点：numpy 标量包装器通过 .item() 归一化后放行；无 numpy 时 SKIP。
# 验证功能：_encode_scalar_arg 的 wrapper 归一化分支。
def check_scalar_numpy_normalization():
    """numpy scalars are accepted via .item() normalization."""
    try:
        import numpy as np
    except ImportError:
        print("SKIP: numpy not installed")
        return
    assert tile_obj._encode_scalar_arg(0, "int64", np.int64(5)) == 5
    assert tile_obj._encode_scalar_arg(0, "float32", np.float32(1.5)) == struct.unpack("<i", struct.pack("<f", 1.5))[0]
    # range check fires after .item() normalization (np.int8(300) itself
    # raises at construction on numpy >= 2.0, so pass a wider wrapper)
    _expect_raises(ValueError, tile_obj._encode_scalar_arg, 0, "int8", np.int64(300))


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
        ubuf_expr=None,
    )
    kernel = stub_kernel(info)
    wrong_dtype = torch.zeros(4, dtype=torch.float32)  # CPU on purpose: dtype must fire first
    _expect_raises(TypeError, kernel._encode_args, [wrong_dtype, 1.0])
    right_dtype = torch.zeros(4, dtype=torch.float16)
    _expect_raises(ValueError, kernel._encode_args, [right_dtype, 1.0])  # CPU -> device error


if __name__ == "__main__":
    sys.exit(run_module(globals()))
