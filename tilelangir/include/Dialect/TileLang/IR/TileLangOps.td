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

include "TileLangDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
    Arg<AnyMemRef, "source memref", [MemRead]>:$src,
    Arg<AnyMemRef, "destination memref", [MemWrite]>:$dest,
    OptionalAttr<I64Attr>:$split_dim,
    OptionalAttr<UnitAttr>:$transpose
  );

  let hasVerifier = 1;

  let assemblyFormat = "$src `,` $dest attr-dict `:` type($src) `to` type($dest)";
}

// ---------------------------------------------------------------------------
// tilelang.gemm
// Matches Python: GemmOp(a, b, c, *, transpose_a/b, clear_accum)
// Generic form: "tilelang.gemm"(%a, %b, %c) {attrs} : (memref, ...) -> ()
// ---------------------------------------------------------------------------
def TileLang_GemmOp : TileLang_Op<"gemm"> {
  let summary = "TileLang matrix multiplication on three memref operands";
  let description = [{
    `tilelang.gemm` takes A, B, and C memref operands. It has no regions.
    Optional attributes retain transpose and accumulator semantics.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "A matrix memref", [MemRead]>:$a,
    Arg<AnyMemRef, "B matrix memref", [MemRead]>:$b,
    Arg<AnyMemRef, "C accumulator memref", [MemRead, MemWrite]>:$c,
    OptionalAttr<UnitAttr>:$transpose_a,
    OptionalAttr<UnitAttr>:$transpose_b,
    OptionalAttr<UnitAttr>:$clear_accum
  );

  let hasVerifier = 1;

  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($a) `,` type($b) `to` type($c)";
}

// ---------------------------------------------------------------------------
// tilelang.scope
// Matches Python: ScopeOp(*, simd_attr=None) with body region
// Generic form: "tilelang.scope"() ({ ... }) {simd_attr = "simd"} : () -> ()
// ---------------------------------------------------------------------------
def TileLang_ScopeOp : TileLang_Op<"scope", [NoTerminator]> {
  let summary = "TileLang frontend scope (e.g. SimdVF)";
  let description = [{
    `tilelang.scope` wraps a body region for a frontend scope such as SimdVF.
    When `simd_attr` is set (for example to `"simd"`), it identifies the scope
    kind for downstream lowering. The body region carries no terminator: the
    Python translator emits it without a `tilelang.scope_yield`, so the
    lowering to `npu.scope` must synthesize the `npu.yield` itself.

    Emitted by the TileLang Python translator for TIRX SBlocks named `SimdVF`.
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$simd_attr);
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;

  let assemblyFormat = "$body attr-dict";
}

// ---------------------------------------------------------------------------
// tilelang.scope_yield
// Terminator for tilelang.scope regions.
// ---------------------------------------------------------------------------
def TileLang_ScopeYieldOp : TileLang_Op<"scope_yield", [
    Terminator, HasParent<"ScopeOp">
  ]> {
  let summary = "Terminator for tilelang.scope";
  let description = [{
    `tilelang.scope_yield` terminates the body region of a `tilelang.scope`.
  }];

  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}


#endif // TILELANG_DIALECT_IR_TILELANGOPS_TD
