#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

python -m pip install mlir-python-bindings -f https://llvm.github.io/eudsl

mkdir -p dist
python -m pip wheel . --no-deps --verbose --wheel-dir dist
