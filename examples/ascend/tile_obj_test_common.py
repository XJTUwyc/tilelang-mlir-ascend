"""Shared helpers for the per-TODO tile_obj test scripts in examples/ascend.

Each TODO in ``tile_obj_load_launch.md`` gets its own test script
(``test_tile_obj_contract.py`` = TODO 3, ``test_tile_obj_dtype.py`` =
TODO 2, ...).  This module holds the pieces every script needs:

- the repo-root import bootstrap (so tilelang imports from the dev tree),
- a minimal ``LaunchInfo`` factory,
- the expect-raises assertion helper,
- a stub ``TileObjKernel`` that skips the FFI lookup in ``__init__``,
- the check-runner / CLI skeleton (``run_module``).

Run the scripts from the repo root with the project venv
(``/home/wuyuchao/dev/.venv``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo importable even when tilelang is not an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tilelang.opentile import tile_obj  # noqa: E402


def _info(name, n_handles=3):
    """A minimal LaunchInfo: ``n_handles`` float16 handle arguments."""
    return tile_obj.LaunchInfo(
        name=name,
        arg_types=["handle"] * n_handles,
        handle_shapes=[],
        handle_dtypes=["float16"] * n_handles,
        grid_exprs=[],
        ubuf_expr=None,
    )


def _expect_raises(exc_type, fn, *args):
    try:
        fn(*args)
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} but nothing was raised")


def stub_kernel(info):
    """A TileObjKernel carrying only ``info`` (no FFI lookup / obj_bytes)."""
    kernel = object.__new__(tile_obj.TileObjKernel)
    kernel.info = info
    return kernel


def _all_checks(module_globals):
    """Collect the zero-arg ``check_*`` functions of one test module."""
    return [
        (name, fn)
        for name, fn in sorted(module_globals.items())
        if name.startswith("check_") and callable(fn) and getattr(fn, "__code__", None) is not None and fn.__code__.co_argcount == 0
    ]


def run_module(module_globals):
    """``--list`` / run-all CLI for one test module; returns an exit code."""
    checks = _all_checks(module_globals)
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        print("Available checks:")
        for name, _ in checks:
            print(f"  {name}")
        return 0
    failed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # the runner reports any failure
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0
