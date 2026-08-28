# TileLang dialect library: registration + op verifiers.

add_mlir_dialect_library(TileLangDialect
  TileLangDialect.cpp
  TileLangOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${TILELANG_DIALECT_SOURCE_DIR}/include/Dialect/TileLang/IR

  DEPENDS
  TileLangDialectIncGen
  TileLangAttrsIncGen
  TileLangOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

# Push include roots to dependents that link TileLangDialect (e.g. tilelang-opt).
# Rooted at include/Dialect so #include "TileLang/IR/..." works.
target_include_directories(TileLangDialect PUBLIC
  $<BUILD_INTERFACE:${TILELANG_DIALECT_SOURCE_DIR}/include/Dialect>
  $<BUILD_INTERFACE:${TILELANG_DIALECT_BINARY_DIR}/include/Dialect>
)
