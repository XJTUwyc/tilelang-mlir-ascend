"""TileLangIR ``tilelang.copy`` split_dim validation tests.

The NPU CopyOp only accepts split_dim 0 (ROW) or 1 (COLUMN); the
translator must reject every other value (e.g. the legacy -1 auto marker
or out-of-range dims such as 2) with a clear error instead of guessing.
Split copies also require rank-2 memrefs; other ranks are rejected.
"""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


def _split_copy_kernel(split_dim, dtype="float32"):
    """Kernel copying a (128, 128) fragment into a (64, 128) fragment."""

    @T.prim_func
    def main(
        A: T.Tensor((128, 128), dtype),
        C: T.Tensor((128, 64), dtype),
    ):
        with T.Kernel(1):
            a_frag = T.alloc_fragment((128, 128), dtype)
            c_frag = T.alloc_fragment((64, 128), dtype)
            T.copy(A, a_frag)
            T.copy(a_frag, c_frag, split_dim=split_dim)

    return main


def _lower_source(kernel):
    artifact = tilelang.lower(kernel, target="tile")
    return artifact.kernel_source


@tilelang.testing.requires_package("mlir")
@pytest.mark.parametrize("split_dim", [0, 1])
def test_split_dim_valid_values_lower(split_dim):
    """split_dim 0 (ROW) and 1 (COLUMN) must lower and reach the IR."""
    source = _lower_source(_split_copy_kernel(split_dim))
    assert "tilelang.copy" in source
    assert f"split_dim = {split_dim}" in source, (
        f"Expected `split_dim = {split_dim}` in the generated MLIR.\n{source}"
    )


@tilelang.testing.requires_package("mlir")
@pytest.mark.parametrize("split_dim", [-1, 2])
def test_split_dim_invalid_values_raise(split_dim):
    """Any value other than 0/1 (e.g. -1 auto marker, out-of-range 2) is rejected."""
    with pytest.raises(ValueError, match="split_dim must be 0 \\(ROW\\) or 1 \\(COLUMN\\)"):
        tilelang.lower(_split_copy_kernel(split_dim), target="tile")


@tilelang.testing.requires_package("mlir")
def test_split_dim_rejects_rank1_memrefs():
    """Rank-1 split copies are rejected; only rank-2 memrefs are supported."""

    @T.prim_func
    def main(
        A: T.Tensor((128,), "float32"),
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_frag = T.alloc_fragment((128,), "float32")
            c_frag = T.alloc_fragment((64,), "float32")
            T.copy(A, a_frag)
            T.copy(a_frag, c_frag, split_dim=0)

    with pytest.raises(ValueError, match="requires rank-2 memrefs"):
        tilelang.lower(main, target="tile")


if __name__ == "__main__":
    tilelang.testing.main()
