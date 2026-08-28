//===- TileLangDialect.td - TileLang dialect definition -------*- tablegen -*-===//
//
// Part of the TileLang project.
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file is consumed by downstream MLIR projects. TileLang itself keeps its
// Python-only dialect registration and does not run mlir-tblgen.

#ifndef TILELANG_DIALECT_IR_TILELANGDIALECT_TD
#define TILELANG_DIALECT_IR_TILELANGDIALECT_TD

include "mlir/IR/OpBase.td"

def TileLang_Dialect : Dialect {
  let name = "tilelang";
  let cppNamespace = "::tilelang";
  let summary = "TileLang tile-programming operations";
  let description = [{
    TileLang operations preserve tile-level programming semantics for
    downstream compiler lowering. Their generic assembly form is the stable
    interchange format emitted by TileLang's Python code generator.
  }];
  let useDefaultAttributePrinterParser = 1;
}

#endif // TILELANG_DIALECT_IR_TILELANGDIALECT_TD
