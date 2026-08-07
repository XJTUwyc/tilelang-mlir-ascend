//===- TileLangOps.h - TileLang operation declarations -----------*- C++ -*-===//
//
// This file declares TileLang operations. It includes the TableGen-generated
// operation declarations (TileLangOps.h.inc) produced by mlir-tblgen from
// TileLangOps.td.
//
//===----------------------------------------------------------------------===//

#ifndef TILELANG_IR_TILELANGOPS_H
#define TILELANG_IR_TILELANGOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// TableGen-generated operation class declarations.
#define GET_OP_CLASSES
#include "TileLang/IR/TileLangOps.h.inc"

#endif // TILELANG_IR_TILELANGOPS_H