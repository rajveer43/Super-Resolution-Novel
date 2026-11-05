"""Tiny YAML config system with `_base_` inheritance and dotted CLI overrides.

Example:
    cfg = load_config("configs/experiments/qg.yaml", ["train.epochs=2", "model.d_model=128"])
    cfg.train.epochs  # -> 2
"""
from __future__ import annotations

import copy
import os
from typing import Any, Iterable

import yaml


class Config(dict):
    """dict with attribute access (recursive)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return _to_plain(self)


def _to_plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_plain(v) for v in x]
    return x


def _wrap(x: Any) -> Any:
    if isinstance(x, dict):
        return Config({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    bases = data.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        merged = deep_merge(merged, _load_yaml(os.path.join(os.path.dirname(path), b)))
    return deep_merge(merged, data)


def _parse_value(s: str) -> Any:
    return yaml.safe_load(s)


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got {ov!r}")
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    return cfg


def load_config(path: str, overrides: Iterable[str] | None = None) -> Config:
    cfg = _load_yaml(path)
    cfg = apply_overrides(cfg, overrides or [])
    return _wrap(cfg)


def save_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(_to_plain(cfg), f, sort_keys=False)
