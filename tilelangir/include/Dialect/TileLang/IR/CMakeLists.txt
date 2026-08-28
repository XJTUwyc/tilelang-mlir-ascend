# TableGen rules for the TileLang dialect and its operations.
# Generated files are written to the matching binary tree location:
#   ${TILELANG_DIALECT_BINARY_DIR}/include/Dialect/TileLang/IR/*.inc
# so they are reachable as #include "TileLang/IR/....inc"
# (include root is .../include/Dialect).

set(LLVM_TARGET_DEFINITIONS TileLangDialect.td)
mlir_tablegen(TileLangDialect.h.inc -gen-dialect-decls -dialect=tilelang)
mlir_tablegen(TileLangDialect.cpp.inc -gen-dialect-defs -dialect=tilelang)
add_public_tablegen_target(TileLangDialectIncGen)

set(LLVM_TARGET_DEFINITIONS TileLangAttrs.td)
mlir_tablegen(TileLangAttrsEnums.h.inc -gen-enum-decls)
mlir_tablegen(TileLangAttrsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TileLangAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=tilelang)
mlir_tablegen(TileLangAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=tilelang)
add_public_tablegen_target(TileLangAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS TileLangOps.td)
mlir_tablegen(TileLangOps.h.inc -gen-op-decls)
mlir_tablegen(TileLangOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TileLangOpsIncGen)
