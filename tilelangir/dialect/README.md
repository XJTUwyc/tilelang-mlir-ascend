# TileLang downstream MLIR dialect

This directory is the source-level contract for downstream MLIR compilers.
TileLang's Python package does not compile or load this dialect: it emits the
same `tilelang.*` operations in generic MLIR assembly.

## Consume from a downstream MLIR project

Add this repository as a third party, then point the downstream project's
TableGen rules at `tilelangir/dialect/include/TileLang/IR/TileLangDialect.td`
and `tilelangir/dialect/include/TileLang/IR/TileLangOps.td`. Generate the
dialect and operation declarations/definitions with that project's
`mlir-tblgen`, and compile the generated code against the same MLIR libraries
as the downstream compiler.

For example, an MLIR CMake project can use:

```cmake
set(LLVM_TARGET_DEFINITIONS
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tilelang/tilelangir/dialect/include/TileLang/IR/TileLangOps.td)
mlir_tablegen(TileLangOps.h.inc -gen-op-decls)
mlir_tablegen(TileLangOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TileLangOpsIncGen)
```

The dialect declaration is generated from `TileLangDialect.td` with
`-gen-dialect-decls` and `-gen-dialect-defs`. Downstream projects own the
resulting C++ library and register `TileLangDialect` before parsing TileLang
IR.

## Compatibility contract

The generic MLIR operation names, region order, and attribute names/types in
`TileLangOps.td` are the public interchange contract. Keep them synchronized
with `tilelangir/_tilelang_ops_gen.py`.

| Operation | Regions | Optional attributes |
| --- | --- | --- |
| `tilelang.copy` | `source`, `dest` | `split_dim: i64`, `transpose: unit` |
| `tilelang.gemm` | `a_region`, `b_region`, `c_region` | `transpose_a: unit`, `transpose_b: unit`, `m: i64`, `n: i64`, `k: i64`, `clear_accum: unit` |
| `tilelang.launch_thread` | `body` | `thread_tag: string`, `extent: i64` |
