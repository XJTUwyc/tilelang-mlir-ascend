"""Compiler-produced final ABI for a Tile object."""

from dataclasses import dataclass
import math


_FLOAT_SCALAR_KINDS = frozenset({"float32", "float64"})
_INTEGER_SCALAR_KINDS = frozenset(
    f"{prefix}{bits}"
    for prefix in ("int", "uint")
    for bits in (8, 16, 32, 64)
)
SUPPORTED_SCALAR_KINDS = _FLOAT_SCALAR_KINDS | _INTEGER_SCALAR_KINDS
_FLOAT32_MAX = 3.4028234663852886e38


def validate_scalar_kind(kind):
    if kind not in SUPPORTED_SCALAR_KINDS:
        raise ValueError(f"unsupported scalar kind: {kind}")


def validate_scalar_value(kind, value):
    """Validate a value for an already validated scalar kind."""
    if kind in _FLOAT_SCALAR_KINDS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{kind} requires a Python numeric scalar")
        try:
            numeric = float(value)
        except OverflowError as exc:
            raise ValueError(f"scalar is outside {kind} range") from exc
        if kind == "float32" and math.isfinite(numeric) and abs(numeric) > _FLOAT32_MAX:
            raise ValueError("scalar is outside float32 range")
        return
    if not isinstance(value, int):
        raise TypeError(f"{kind} requires a Python integer")
    unsigned = kind.startswith("uint")
    bits = int(kind[4:] if unsigned else kind[3:])
    lo, hi = (0, 2**bits - 1) if unsigned else (-2 ** (bits - 1), 2 ** (bits - 1) - 1)
    if not lo <= value <= hi:
        raise ValueError(f"{value} is outside {kind} range")
    # TileObjectKernel currently transports scalar integers through signed int64.
    if kind == "uint64" and value > 2**63 - 1:
        raise ValueError("uint64 above INT64_MAX requires a launcher encoding extension")


def validate_scalar(kind, value):
    validate_scalar_kind(kind)
    validate_scalar_value(kind, value)


@dataclass(frozen=True)
class ParameterBinding:
    index: int  # Index in artifact.params, INCLUDING output parameters.

    def __post_init__(self):
        if type(self.index) is not int or self.index < 0:
            raise ValueError("parameter index must be a non-negative integer")


@dataclass(frozen=True)
class ConstantBinding:
    value: int | float | bool


@dataclass(frozen=True)
class DeviceArgument:
    kind: str
    source: ParameterBinding | ConstantBinding

    def __post_init__(self):
        if not isinstance(self.source, (ParameterBinding, ConstantBinding)):
            raise TypeError("source must be a ParameterBinding or ConstantBinding")
        if self.kind == "handle":
            if isinstance(self.source, ConstantBinding):
                raise ValueError("handle constants are unsupported; pass a tensor parameter")
        elif isinstance(self.source, ConstantBinding):
            validate_scalar(self.kind, self.source.value)
        else:
            validate_scalar_kind(self.kind)


@dataclass(frozen=True)
class TileLaunchSpec:
    kernel_name: str
    arguments: tuple[DeviceArgument, ...]
    block_count: int
    dynamic_ubuf_bytes: int = 0

    def __post_init__(self):
        if not isinstance(self.kernel_name, str) or not self.kernel_name or "\x00" in self.kernel_name:
            raise ValueError("kernel_name must be a non-empty runtime name without NUL")
        object.__setattr__(self, "arguments", tuple(self.arguments))
        if not all(isinstance(arg, DeviceArgument) for arg in self.arguments):
            raise TypeError("arguments must contain DeviceArgument values")
        if type(self.block_count) is not int or not 0 < self.block_count <= 2**32 - 1:
            raise ValueError("block_count must be a positive uint32 value")
        if type(self.dynamic_ubuf_bytes) is not int or not 0 <= self.dynamic_ubuf_bytes <= 2**32 - 1:
            raise ValueError("dynamic_ubuf_bytes must be a non-negative uint32 byte count")
