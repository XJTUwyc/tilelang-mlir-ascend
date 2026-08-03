"""Scope constructs exposed by the TileLang language surface."""

from tvm.tirx.script.builder import frame
from tvm.tirx.script.builder.ir import sblock


def SimdVF() -> frame.SBlockFrame:  # pylint: disable=invalid-name
    """Create a SIMD vector-function scope in Core TIRX."""
    return sblock("SimdVF")
