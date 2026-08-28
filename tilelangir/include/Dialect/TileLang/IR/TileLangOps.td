//===- TileLangOps.td - TileLang operation definitions --------*- tablegen -*-===//
//
// Part of the TileLang project.
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Keep this file synchronized with the matching repository copy.
// TileLang's hand-written Python OpView classes follow the same operation,
// region, and attribute contract.
//
//===----------------------------------------------------------------------===//

#ifndef TILELANG_DIALECT_IR_TILELANGOPS_TD
#define TILELANG_DIALECT_IR_TILELANGOPS_TD

include "TileLang/IR/TileLangAttrs.td"
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
// Matches Python: ScopeOp(*, mode, threads=None) with body region
// Generic form: "tilelang.scope"() ({ ... })
//               {mode = #tilelang.scope_mode<simt>,
//                threads = 128 : ui32}
//               : () -> ()
// ---------------------------------------------------------------------------
def TileLang_ScopeOp : TileLang_Op<"scope", [NoTerminator]> {
  let summary = "TileLang frontend execution scope";
  let description = [{
    `tilelang.scope` wraps a body region for a frontend execution scope. The
    required `mode` is one of `default`, `simd`, or `simt`.

    In SIMT mode, `threads` is the requested physical worker count. Logical
    work is represented independently by parallel loops inside the region.
  }];

  let arguments = (ins
    TileLang_ScopeModeAttr:$mode,
    OptionalAttr<UI32Attr>:$threads
  );
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
