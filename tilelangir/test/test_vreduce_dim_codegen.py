"""TileLangIR ``tl.tileop.vreduce_*`` explicit ``dim`` argument tests.

Verifies that ``T.vreduce_max`` / ``T.vreduce_sum`` accept an optional ``dim``
argument naming the reduced source dimensions, with negative indices following
Python semantics, and that omitting ``dim`` still falls back to shape inference.
These are pure IR codegen tests: they call ``tilelang.lower(..., target="tile")``
and assert on the generated MLIR source string.  No Ascend hardware required.
"""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


def _reduce_kernel(
    dst_shape,
    dim,
    op_name="vreduce_max",
    M: int = 128,
    N: int = 128,
    dtype: str = "float32",
):
    """Reduce a (M, N) fragment to ``dst_shape`` with an optional explicit dim."""

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor(tuple(dst_shape), dtype),
    ):
        with T.Kernel(1) as bx:
            with T.SimdVF():
                a_frag = T.alloc_frag((M, N), dtype)
                dst_frag = T.alloc_frag(tuple(dst_shape), dtype)
                T.copy(A[0:M, 0:N], a_frag)

                reduce_op = getattr(T, op_name)
                if dim is None:
                    reduce_op(a_frag, dst_frag)
                else:
                    reduce_op(a_frag, dst_frag, dim=dim)

                T.copy(dst_frag, C)

    return main


def _reduce_3d_kernel(
    dim,
    op_name="vreduce_sum",
    B: int = 4,
    M: int = 8,
    N: int = 16,
    dtype: str = "float32",
):
    """Reduce a (B, M, N) fragment along two dims to a (M,) result."""

    @T.prim_func
    def main(
        A: T.Tensor((B, M, N), dtype),
        C: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1) as bx:
            with T.SimdVF():
                a_frag = T.alloc_frag((B, M, N), dtype)
                dst_frag = T.alloc_frag((M,), dtype)
                T.copy(A[0:B, 0:M, 0:N], a_frag)

                reduce_op = getattr(T, op_name)
                reduce_op(a_frag, dst_frag, dim=dim)

                T.copy(dst_frag, C)

    return main


def _kernel_source(artifact) -> str:
    source = getattr(artifact, "kernel_source", None)
    if source:
        return str(source)
    device_mod = getattr(artifact, "device_mod", None)
    if device_mod is not None and hasattr(device_mod, "inspect_source"):
        inspected = device_mod.inspect_source()
        if inspected:
            return str(inspected)
    raise AssertionError("lower(target='tile') did not produce inspectable MLIR source")


@tilelang.testing.requires_package("mlir")
@pytest.mark.parametrize("op_name", ["vreduce_max", "vreduce_sum"])
def test_reduce_dim_last(op_name):
    """dim=1 on (M, N) -> (M,): row reduce, dimensions = [1]."""
    artifact = tilelang.lower(_reduce_kernel([128], 1, op_name), target="tile")
    mlir = _kernel_source(artifact)
    assert "linalg.reduce" in mlir
    assert "dimensions = [1]" in mlir


@tilelang.testing.requires_package("mlir")
@pytest.mark.parametrize("op_name", ["vreduce_max", "vreduce_sum"])
def test_reduce_dim_first(op_name):
    """dim=0 on (M, N) -> (N,): column reduce, dimensions = [0]."""
    artifact = tilelang.lower(_reduce_kernel([128], 0, op_name), target="tile")
    mlir = _kernel_source(artifact)
    assert "dimensions = [0]" in mlir


@tilelang.testing.requires_package("mlir")
@pytest.mark.parametrize("op_name", ["vreduce_max", "vreduce_sum"])
def test_reduce_dim_negative(op_name):
    """dim=-1 is equivalent to dim=1."""
    artifact = tilelang.lower(_reduce_kernel([128], -1, op_name), target="tile")
    mlir = _kernel_source(artifact)
    assert "dimensions = [1]" in mlir


@tilelang.testing.requires_package("mlir")
@pytest.mark.parametrize("op_name", ["vreduce_max", "vreduce_sum"])
def test_reduce_dim_inferred_when_omitted(op_name):
    """No dim: shape inference keeps working (trailing dim reduced)."""
    artifact = tilelang.lower(_reduce_kernel([128], None, op_name), target="tile")
    mlir = _kernel_source(artifact)
    assert "dimensions = [1]" in mlir


@tilelang.testing.requires_package("mlir")
@pytest.mark.parametrize("op_name", ["vreduce_max", "vreduce_sum"])
def test_reduce_dim_tuple(op_name):
    """dim=(0, 2) on (B, M, N) -> (M,): dimensions = [0, 2]."""
    artifact = tilelang.lower(_reduce_3d_kernel((0, 2), op_name), target="tile")
    mlir = _kernel_source(artifact)
    assert "dimensions = [0, 2]" in mlir


@tilelang.testing.requires_package("mlir")
@pytest.mark.parametrize("op_name", ["vreduce_max", "vreduce_sum"])
def test_reduce_dim_shape_mismatch(op_name):
    """dim=0 would produce (N,) but dst is (M,): must raise."""
    with pytest.raises(ValueError, match="destination shape"):
        tilelang.lower(_reduce_kernel([128], 0, op_name, M=128, N=64), target="tile")


@tilelang.testing.requires_package("mlir")
@pytest.mark.parametrize("op_name", ["vreduce_max", "vreduce_sum"])
def test_reduce_dim_out_of_range(op_name):
    with pytest.raises(ValueError, match="out of range"):
        tilelang.lower(_reduce_kernel([128], 5, op_name), target="tile")


@tilelang.testing.requires_package("mlir")
@pytest.mark.parametrize("op_name", ["vreduce_max", "vreduce_sum"])
def test_reduce_dim_duplicate(op_name):
    with pytest.raises(ValueError, match="Duplicate"):
        tilelang.lower(_reduce_kernel([128], (1, 1), op_name), target="tile")


def test_reduce_dim_dsl_validation():
    from tilelang.language.simd.vector import vreduce_max

    with pytest.raises(TypeError, match="Unsupported dim type"):
        vreduce_max(None, None, dim=1.5)
    with pytest.raises(ValueError, match="must not be empty"):
        vreduce_max(None, None, dim=())


if __name__ == "__main__":
    tilelang.testing.main()
