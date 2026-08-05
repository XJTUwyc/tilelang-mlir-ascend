//===- TileLangOps.td - TileLang operation definitions --------*- tablegen -*-===//
//
// Part of the TileLang project.
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Keep this file synchronized with tilelangir/_tilelang_ops_gen.py. The Python
// hand-written OpView classes are the codegen path; this TableGen file is the
// downstream C++ dialect contract for the same operation names, operands,
// regions, and attributes.
//
//===----------------------------------------------------------------------===//

#ifndef TILELANG_DIALECT_IR_TILELANGOPS_TD
#define TILELANG_DIALECT_IR_TILELANGOPS_TD

include "TileLang/IR/TileLangDialect.td"
include "mlir/IR/OpBase.td"

class TileLang_Op<string mnemonic, list<Trait> traits = []>
    : Op<TileLang_Dialect, mnemonic, traits>;

// ---------------------------------------------------------------------------
// tilelang.copy
// Matches Python: CopyOp(src, dest, *, split_dim=None, transpose=None)
// Generic form: "tilelang.copy"(%src, %dest) {attrs} : (memref, memref) -> ()
// ---------------------------------------------------------------------------
def TileLang_CopyOp : TileLang_Op<"copy"> {
  let summary = "Copy values between two memref SSA operands";
  let description = [{
    `tilelang.copy` takes a source memref and a destination memref. It has no
    regions. Optional attributes retain frontend annotations such as split and
    transpose.

    Emitted by the TileLang Python translator for `tl.tileop.copy` after
    lowering each side to a memref view (or allocation).
  }];

  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dest,
    OptionalAttr<I64Attr>:$split_dim,
    OptionalAttr<UnitAttr>:$transpose
  );

  // No results, no regions (parity with _tilelang_ops_gen.CopyOp._ODS_REGIONS).
}

// ---------------------------------------------------------------------------
// tilelang.gemm
// Matches Python: GemmOp(a, b, c, *, transpose_a/b, m, n, k, clear_accum)
// Generic form: "tilelang.gemm"(%a, %b, %c) {attrs} : (memref, ...) -> ()
// ---------------------------------------------------------------------------
def TileLang_GemmOp : TileLang_Op<"gemm"> {
  let summary = "TileLang matrix multiplication on three memref operands";
  let description = [{
    `tilelang.gemm` takes A, B, and C memref operands. It has no regions.
    Optional attributes retain frontend matrix dimensions and
    transpose/accumulator semantics.
  }];

  let arguments = (ins
    AnyMemRef:$a,
    AnyMemRef:$b,
    AnyMemRef:$c,
    OptionalAttr<UnitAttr>:$transpose_a,
    OptionalAttr<UnitAttr>:$transpose_b,
    OptionalAttr<I64Attr>:$m,
    OptionalAttr<I64Attr>:$n,
    OptionalAttr<I64Attr>:$k,
    OptionalAttr<UnitAttr>:$clear_accum
  );
}

// ---------------------------------------------------------------------------
// tilelang.scope
// Matches Python: ScopeOp(*, simd_attr=None) with body region
// Generic form: "tilelang.scope"() ({ ... }) {simd_attr = "simd"} : () -> ()
// ---------------------------------------------------------------------------
def TileLang_ScopeOp : TileLang_Op<"scope"> {
  let summary = "TileLang frontend scope (e.g. SimdVF)";
  let description = [{
    `tilelang.scope` wraps a body region for a frontend scope such as SimdVF.
    When `simd_attr` is set (for example to `"simd"`), it identifies the scope
    kind for downstream lowering.

    Emitted by the TileLang Python translator for TIRX SBlocks named `SimdVF`.
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$simd_attr);
  let regions = (region AnyRegion:$body);
}

// Note: `tilelang.launch_thread` is intentionally absent. For the current
// tile backend, only `blockIdx.x` is effective and is modeled as an i32
// function argument plus a `BlockIdx` function attribute rather than a custom
// operation.

#endif // TILELANG_DIALECT_IR_TILELANGOPS_TD
