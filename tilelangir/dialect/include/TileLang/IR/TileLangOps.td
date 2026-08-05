//===- TileLangOps.td - TileLang operation definitions --------*- tablegen -*-===//
//
// Part of the TileLang project.
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TILELANG_DIALECT_IR_TILELANGOPS_TD
#define TILELANG_DIALECT_IR_TILELANGOPS_TD

include "TileLang/IR/TileLangDialect.td"

class TileLang_Op<string mnemonic, list<Trait> traits = []>
    : Op<TileLang_Dialect, mnemonic, traits>;

def TileLang_CopyOp : TileLang_Op<"copy"> {
  let summary = "Copy values between TileLang regions";
  let description = [{
    `tilelang.copy` contains a source region and a destination region. Both
    regions are intentionally unconstrained so a downstream compiler can
    preserve frontend semantics before selecting its own memory operations.
  }];

  let arguments = (ins
    OptionalAttr<I64Attr>:$split_dim,
    OptionalAttr<UnitAttr>:$transpose
  );
  let regions = (region
    AnyRegion:$source,
    AnyRegion:$dest
  );
}

def TileLang_GemmOp : TileLang_Op<"gemm"> {
  let summary = "TileLang matrix multiplication";
  let description = [{
    `tilelang.gemm` contains A, B, and C regions. Its optional attributes
    retain frontend matrix dimensions and transpose/accumulator semantics.
  }];

  let arguments = (ins
    OptionalAttr<UnitAttr>:$transpose_a,
    OptionalAttr<UnitAttr>:$transpose_b,
    OptionalAttr<I64Attr>:$m,
    OptionalAttr<I64Attr>:$n,
    OptionalAttr<I64Attr>:$k,
    OptionalAttr<UnitAttr>:$clear_accum
  );
  let regions = (region
    AnyRegion:$a_region,
    AnyRegion:$b_region,
    AnyRegion:$c_region
  );
}

def TileLang_LaunchThreadOp : TileLang_Op<"launch_thread"> {
  let summary = "TileLang thread binding";
  let description = [{
    `tilelang.launch_thread` retains a frontend thread binding and its body.
    `thread_tag` uses the frontend spelling, such as `blockIdx.x`.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$thread_tag,
    OptionalAttr<I64Attr>:$extent
  );
  let regions = (region AnyRegion:$body);
}

#endif // TILELANG_DIALECT_IR_TILELANGOPS_TD
