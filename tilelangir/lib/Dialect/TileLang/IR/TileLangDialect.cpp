//===- TileLangDialect.cpp - TileLang dialect registration ------*- C++ -*-===//
//
// Hand-written dialect registration. The dialect class skeleton is generated
// by mlir-tblgen from TileLangDialect.td and included via TileLangDialect.h.
// This file provides the dialect's initialize() method which registers all
// operations.
//
//===----------------------------------------------------------------------===//

#include "TileLang/IR/TileLangDialect.h"
#include "TileLang/IR/TileLangAttrs.h"
#include "TileLang/IR/TileLangOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace tilelang;

// TableGen-generated dialect definitions (e.g. TileLangDialect constructor).
#include "TileLang/IR/TileLangAttrsEnums.cpp.inc"
#include "TileLang/IR/TileLangDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TileLang dialect initialization
//===----------------------------------------------------------------------===//

void TileLangDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TileLang/IR/TileLangAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "TileLang/IR/TileLangOps.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "TileLang/IR/TileLangAttrs.cpp.inc"
