# TileLang downstream MLIR dialect

This directory holds public headers and ODS for the TileLang dialect. The
build contract lives at `tilelangir/CMakeLists.txt`. TileLang's Python package
does not compile or load this dialect: it emits the same `tilelang.*`
operations in generic MLIR assembly via `tilelangir/_tilelang_ops_gen.py`.

TableGen sources and public headers live under `include/Dialect/TileLang/IR/`;
implementation lives under `lib/Dialect/TileLang/IR/`.

## Directory layout

```
tilelangir/
  CMakeLists.txt                              # build contract
  include/
    README.md                                 # this file
    Dialect/TileLang/IR/
      CMakeLists.txt                          # TableGen → TileLangDialect/Ops *.inc
      TileLangDialect.td / .h
      TileLangOps.td / .h
  lib/Dialect/TileLang/IR/
    CMakeLists.txt                            # add_mlir_dialect_library(TileLangDialect)
    TileLangDialect.cpp
    TileLangOps.cpp
```

| Path | Role |
| ---- | ---- |
| `include/Dialect/TileLang/IR/` | Public headers (`.h`) and ODS (`.td`); TableGen produces `.inc` under the binary tree |
| `lib/Dialect/TileLang/IR/` | Hand-written dialect registration and op verifiers; builds target `TileLangDialect` |
| `tilelangir/CMakeLists.txt` | Sets include roots; `add_subdirectory(include/Dialect/TileLang/IR)` then `add_subdirectory(lib/Dialect/TileLang/IR)` |

Include roots are `include/Dialect` (source) and the matching binary path, so
consumers write:

```c++
#include "TileLang/IR/TileLangDialect.h"
#include "TileLang/IR/TileLangOps.h"
```

## Consume from a downstream MLIR project

### Preferred: `add_subdirectory`

After `find_package(MLIR)` and loading `TableGen` / `AddLLVM` / `AddMLIR`,
add the `tilelangir` directory as a subproject:

```cmake
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tilelang/tilelangir
  tilelang_dialect)

# then link the dialect into your tool/pass library
target_link_libraries(my-opt PRIVATE TileLangDialect)
```

That builds the TableGen targets (`TileLangDialectIncGen`, `TileLangOpsIncGen`)
and the `TileLangDialect` library. Dependents that link `TileLangDialect`
get the source and generated include roots via the target interface.

A working consumer of this path lives in `tilelangir/dialect_verify/`.

### Alternative: own TableGen rules

Point the downstream project's TableGen at the same `.td` files, generate
declarations/definitions with that project's `mlir-tblgen`, and compile against
the same MLIR libraries as the downstream compiler:

```cmake
set(LLVM_TARGET_DEFINITIONS
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tilelang/tilelangir/include/Dialect/TileLang/IR/TileLangOps.td)
mlir_tablegen(TileLangOps.h.inc -gen-op-decls)
mlir_tablegen(TileLangOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TileLangOpsIncGen)
```

The dialect declaration is generated from `TileLangDialect.td` with
`-gen-dialect-decls` and `-gen-dialect-defs` (optionally `-dialect=tilelang`).
You still need to compile `lib/Dialect/TileLang/IR/*.cpp` (or equivalent
registration code) and register `TileLangDialect` before parsing TileLang IR.

## Compatibility contract

The generic MLIR operation names, operand order/types, region order, and
attribute names/types in `TileLangOps.td` are the public interchange contract.
**Keep them synchronized with** `tilelangir/_tilelang_ops_gen.py`**.**

| Operation        | Operands                                       | Regions | Optional attributes                                       |
| ---------------- | ---------------------------------------------- | ------- | --------------------------------------------------------- |
| `tilelang.copy`  | `src: AnyMemRef`, `dest: AnyMemRef`            | none    | `split_dim: i64`, `transpose: unit`                       |
| `tilelang.gemm`  | `a: AnyMemRef`, `b: AnyMemRef`, `c: AnyMemRef` | none    | `transpose_a: unit`, `transpose_b: unit`, `clear_accum: unit` |
| `tilelang.scope` | none                                           | `body`  | `simd_attr: string`                                       |
