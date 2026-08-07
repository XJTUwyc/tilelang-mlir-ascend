# TileLang Dialect CMake
# This file is the build contract for downstream MLIR projects.
# It provides:
#   1. TableGen targets under include/Dialect/TileLang/IR (generate C++ from .td)
#   2. A dialect library under lib/Dialect/TileLang/IR
#      (TileLangDialect + hand-written verifiers)
#
# Layout:
#   tilelangir/
#     include/Dialect/TileLang/IR/   # .td / .h + TableGen CMake
#     lib/Dialect/TileLang/IR/       # .cpp + library CMake
#
# Public #include paths are rooted at include/Dialect, so consumers write:
#   #include "TileLang/IR/TileLangOps.h"
#
# Downstream projects can either:
#   a) add_subdirectory() this directory to get the TileLangDialect target, or
#   b) reference the .td files directly with their own mlir-tblgen rules.

cmake_minimum_required(VERSION 3.20.0)

set(TILELANG_DIALECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TILELANG_DIALECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Include roots so #include "TileLang/IR/..." resolves for both source headers
# and TableGen-generated .inc files (produced under binary include/Dialect/).
include_directories(
  ${TILELANG_DIALECT_SOURCE_DIR}/include/Dialect
  ${TILELANG_DIALECT_BINARY_DIR}/include/Dialect
)

# TableGen (headers) then library (implementation).
add_subdirectory(include/Dialect/TileLang/IR)
add_subdirectory(lib/Dialect/TileLang/IR)
