"""Vector operations represented as Core TIRX tile calls."""

from __future__ import annotations

from typing import Literal

from tvm import tirx

from tilelang._typing import BufferLikeType, DType, PyPrimExpr
from tilelang.utils.language import to_tile_region


AccessType = Literal["r", "w", "rw"]
OperandType = BufferLikeType | PyPrimExpr


def _normalize_operand(
    value: OperandType,
    *,
    access_type: AccessType | None = None,
) -> tirx.PrimExpr:
    """Normalize a Python SIMD operand into a Core TIRX expression.

    Buffer and BufferRegion operands require an access type and become
    ``tl.tileop.region`` calls.  A BufferLoad becomes a region when an access
    type is supplied, or remains a scalar expression otherwise.  Existing
    PrimExpr values are preserved and Python scalar values become immediates.
    """
    if isinstance(value, (tirx.Buffer, tirx.BufferRegion)):
        if access_type is None:
            raise TypeError("A Buffer or BufferRegion SIMD operand requires an access type")
        return to_tile_region(value, access_type=access_type)

    # BufferLoad is a PrimExpr, so it must be handled before the general
    # PrimExpr case.  Its role determines whether it denotes a scalar value or
    # a one-element/read-write region.
    if isinstance(value, tirx.BufferLoad):
        if access_type is None:
            return value
        return to_tile_region(value, access_type=access_type)

    if isinstance(value, tirx.PrimExpr):
        return value

    if isinstance(value, (bool, int, float)):
        if access_type in ("w", "rw"):
            raise TypeError("A writable SIMD operand cannot be a Python scalar")
        return tirx.const(value)

    raise TypeError(f"Unsupported SIMD operand type: {type(value).__name__}")


def _call_vector_op(op_name: str, *args: tirx.PrimExpr) -> tirx.PrimExpr:
    """Create an opaque vector TileOp call for direct backend codegen."""
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get(f"tl.tileop.{op_name}"),
        *args,
    )


def _binary_vector_op(
    op_name: str,
    src0: OperandType,
    src1: OperandType,
    dst: OperandType,
) -> tirx.PrimExpr:
    return _call_vector_op(
        op_name,
        _normalize_operand(src0, access_type="r"),
        _normalize_operand(src1, access_type="r"),
        _normalize_operand(dst, access_type="w"),
    )


def _normalize_reduce_dims(dim) -> list[int]:
    """Normalize a ``dim`` argument into a list of ints (negative kept for the translator)."""
    if isinstance(dim, int):
        dims = [dim]
    elif isinstance(dim, (tuple, list)):
        dims = [int(d) for d in dim]
    else:
        raise TypeError(f"Unsupported dim type: {type(dim).__name__}, expected int or sequence of int")
    if len(dims) == 0:
        raise ValueError("dim must not be empty")
    return dims


def _reduce_vector_op(
    op_name: str,
    src: OperandType,
    dst: OperandType,
    dim: int | tuple[int, ...] | list[int] | None = None,
) -> tirx.PrimExpr:
    # Validate dim before touching the operands so bad dim values report
    # the dim error rather than an operand error.
    dims = _normalize_reduce_dims(dim) if dim is not None else []
    args = [
        _normalize_operand(src, access_type="r"),
        _normalize_operand(dst, access_type="w"),
    ]
    args.extend(tirx.const(d, "int64") for d in dims)
    return _call_vector_op(op_name, *args)


def vmuls(src: OperandType, scalar: PyPrimExpr, dst: OperandType) -> tirx.PrimExpr:
    """Multiply a vector region by a scalar and write it to a vector region."""
    return _call_vector_op(
        "vmuls",
        _normalize_operand(src, access_type="r"),
        _normalize_operand(scalar),
        _normalize_operand(dst, access_type="w"),
    )


def vadd(src0: OperandType, src1: OperandType, dst: OperandType) -> tirx.PrimExpr:
    """Add two vector regions and write the result to a vector region."""
    return _binary_vector_op("vadd", src0, src1, dst)


def vmul(src0: OperandType, src1: OperandType, dst: OperandType) -> tirx.PrimExpr:
    """Multiply two vector regions and write the result to a vector region."""
    return _binary_vector_op("vmul", src0, src1, dst)


def vsub(src0: OperandType, src1: OperandType, dst: OperandType) -> tirx.PrimExpr:
    """Subtract two vector regions and write the result to a vector region."""
    return _binary_vector_op("vsub", src0, src1, dst)


def vmax(src0: OperandType, src1: OperandType, dst: OperandType) -> tirx.PrimExpr:
    """Take the elementwise maximum and write it to a vector region."""
    return _binary_vector_op("vmax", src0, src1, dst)


def vreduce_max(
    src: OperandType,
    dst: OperandType,
    dim: int | tuple[int, ...] | list[int] | None = None,
) -> tirx.PrimExpr:
    """Reduce a vector region by maximum into a destination region.

    ``dim`` names the reduced (eliminated) source dimensions.  Negative indices
    follow Python semantics.  When omitted the dimensions are inferred by
    comparing the source and destination shapes.
    """
    return _reduce_vector_op("vreduce_max", src, dst, dim)


def vreduce_sum(
    src: OperandType,
    dst: OperandType,
    dim: int | tuple[int, ...] | list[int] | None = None,
) -> tirx.PrimExpr:
    """Reduce a vector region by summation into a destination region.

    ``dim`` names the reduced (eliminated) source dimensions.  Negative indices
    follow Python semantics.  When omitted the dimensions are inferred by
    comparing the source and destination shapes.
    """
    return _reduce_vector_op("vreduce_sum", src, dst, dim)


def vexp(src: OperandType, dst: OperandType) -> tirx.PrimExpr:
    """Apply the exponential function elementwise to a vector region."""
    return _call_vector_op(
        "vexp",
        _normalize_operand(src, access_type="r"),
        _normalize_operand(dst, access_type="w"),
    )


def vexpdif(src0: OperandType, src1: OperandType, dst: OperandType) -> tirx.PrimExpr:
    """Preserve a vector exponential-difference operation in Core TIRX."""
    return _binary_vector_op("vexpdif", src0, src1, dst)


def vcvt(src: OperandType, dst: OperandType, dtype: DType) -> tirx.PrimExpr:
    """Preserve a vector conversion and its destination dtype in Core TIRX."""
    return _call_vector_op(
        "vcvt",
        _normalize_operand(src, access_type="r"),
        _normalize_operand(dst, access_type="w"),
        tirx.StringImm(str(dtype)),
    )
