#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

export USE_CUDA="${USE_CUDA:-0}"

echo "===== Step 1: Fetching Git submodules ====="
git submodule sync --recursive
git submodule update --init --recursive

echo "===== Step 2: Installing Python build and runtime dependencies ====="
python -m pip install \
    --requirement requirements.txt \
    --requirement requirements-dev.txt \
    mlir-python-bindings \
    --find-links https://llvm.github.io/eudsl

echo "===== Step 3: Installing tilelang in editable mode (USE_CUDA=${USE_CUDA}) ====="
python -m pip install \
    --editable . \
    --verbose \
    --no-deps \
    --no-build-isolation

echo "===== Step 4: Building wheel into dist/ (USE_CUDA=${USE_CUDA}) ====="
mkdir -p dist
python -m pip wheel \
    . \
    --no-deps \
    --verbose \
    --no-build-isolation \
    --wheel-dir dist

echo "===== Step 5: Verifying editable import and wheel output ====="
(
    cd /
    TILELANG_REPO_ROOT="${repo_root}" python - <<'PY'
import os
from pathlib import Path

import tilelang
import tilelangir

repo_root = Path(os.environ["TILELANG_REPO_ROOT"]).resolve()
for package in (tilelang, tilelangir):
    package_file = Path(package.__file__).resolve()
    if repo_root not in package_file.parents:
        raise RuntimeError(
            f"{package.__name__} is not imported from the editable source tree: "
            f"{package_file}"
        )
    print(f"{package.__name__}: {package_file}")
PY
)

wheel_count="$(find dist -maxdepth 1 -type f -name 'tilelang-*.whl' | wc -l)"
if [ "${wheel_count}" -eq 0 ]; then
    echo "Error: no TileLang wheel was produced in ${repo_root}/dist" >&2
    exit 1
fi

find dist -maxdepth 1 -type f -name 'tilelang-*.whl' -print
echo "Done."
