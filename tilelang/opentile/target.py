from __future__ import annotations

from collections.abc import Mapping

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_normalizer


_K_DL_EXT_DEV = 12


def target_is_tile(target: Target) -> bool:
    return "tile" in target.keys


def _make_tile_target(target_dict: Mapping[str, object] | None = None) -> Target:
    config = dict(target_dict or {})
    keys = config.get("keys", ())
    if isinstance(keys, str):
        keys = (keys,)
    config["kind"] = "c"
    config["keys"] = list(dict.fromkeys([*keys, "tile"]))
    config.setdefault("target_device_type", _K_DL_EXT_DEV)
    return Target(config)


def _with_tile_key(target: Target) -> Target:
    return _make_tile_target(target.export())


def normalize_tile_target(target: TargetLike) -> Target | None:
    if isinstance(target, Target):
        return _with_tile_key(target) if target_is_tile(target) else None

    if isinstance(target, Mapping):
        config = dict(target)
        keys = config.get("keys", ())
        if isinstance(keys, str):
            keys = (keys,)
        if config.get("kind") == "tile" or "tile" in keys:
            return _make_tile_target(config)
        return None

    if isinstance(target, str):
        normalized = target.strip()
        if normalized in {"tile", "c -keys=tile"}:
            return _make_tile_target()
        try:
            parsed = Target(normalized)
        except Exception:
            return None
        return _with_tile_key(parsed) if target_is_tile(parsed) else None

    return None


register_target_normalizer("tile", normalize_tile_target, override=True)
