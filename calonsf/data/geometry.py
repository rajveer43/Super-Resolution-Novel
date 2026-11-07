"""Detector geometry, multi-resolution scale specs, and the next-scale ladder.

Every dataset is described as a list of *layer groups*. A group is a stack of `n_layers` 2D grids that
share one shape (H, W):
    Quark-Gluon  : 1 group  (3 layers = Tracks/ECAL/HCAL, 128x128 after padding 125->128)
    CaloGAN      : 3 groups (layer_0 3x96, layer_1 12x12, layer_2 12x6) - heterogeneous granularity
    CaloChallenge: 1 group  (45 longitudinal layers, each a 16 (alpha) x 9 (r) polar grid)

Each cell gets continuous coordinates (depth, u, v). Non-periodic axes live in [-1, 1]; periodic axes
(CaloChallenge alpha) are in radians [0, 2pi). This is what makes the model resolution-agnostic
(the cell-set view comes from the Graph Denoising paper, coordinates + RoPE from V-JEPA 2).

A `ScaleSpec` is one resolution of the whole detector (a shape per group). Cells of all groups are
concatenated into a "dense" vector of length T, which we use for scatter/gather.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

TWO_PI = 2.0 * math.pi


@dataclass
class LayerGroup:
    name: str
    layer_names: list[str]
    shape: tuple[int, int]  # native (ground-truth) resolution
    depths: list[float]  # depth coordinate per layer, in [-1, 1]
    periodic: tuple[bool, bool] = (False, False)
    sr_axes: tuple[bool, bool] = (True, True)  # which axes get super-resolved

    @property
    def n_layers(self) -> int:
        return len(self.layer_names)


@dataclass
class Geometry:
    name: str
    groups: list[LayerGroup]
    energy_scale: float = 1.0  # E_scale in log1p(E / E_scale)
    # optional physical description used by physics observables
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.group_names = [g.name for g in self.groups]
        off = 0
        self.layer_offset = {}
        for g in self.groups:
            self.layer_offset[g.name] = off
            off += g.n_layers
        self.n_layers_total = off
        self.depths_all = [d for g in self.groups for d in g.depths]
        per = {g.periodic for g in self.groups}
        if len(per) != 1:
            raise ValueError("All groups must share periodicity flags (needed for a shared RoPE).")
        self.periodic = self.groups[0].periodic

    def group(self, name: str) -> LayerGroup:
        return self.groups[self.group_names.index(name)]

    def native_shapes(self) -> dict[str, tuple[int, int]]:
        return {g.name: tuple(g.shape) for g in self.groups}

    def spec(self, shapes: dict[str, tuple[int, int]]) -> "ScaleSpec":
        key = tuple(tuple(shapes[g]) for g in self.group_names)
        return _spec_cached(self, key)

    def native_spec(self) -> "ScaleSpec":
        return self.spec(self.native_shapes())

    def __hash__(self) -> int:  # used by lru_cache
        return id(self)


@lru_cache(maxsize=256)
def _spec_cached(geom: Geometry, key: tuple) -> "ScaleSpec":
    return ScaleSpec(geom, {g: tuple(s) for g, s in zip(geom.group_names, key)})


class ScaleSpec:
    """One resolution of the detector. Provides dense<->grid conversion and analytic cell info."""

    def __init__(self, geom: Geometry, shapes: dict[str, tuple[int, int]]):
        self.geom = geom
        self.shapes = shapes
        self.sizes = [geom.group(g).n_layers * shapes[g][0] * shapes[g][1] for g in geom.group_names]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)]).astype(np.int64)
        self.T = int(self.offsets[-1])
        self._dev_cache: dict = {}

    # ------------------------------------------------------------------ grids <-> dense
    def to_dense(self, grids: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for g in self.geom.group_names:
            x = grids[g]
            assert tuple(x.shape[-2:]) == tuple(self.shapes[g]), (g, x.shape, self.shapes[g])
            parts.append(x.reshape(x.shape[0], -1))
        return torch.cat(parts, dim=1)

    def to_grids(self, dense: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        B = dense.shape[0]
        for i, g in enumerate(self.geom.group_names):
            n = self.geom.group(g).n_layers
            H, W = self.shapes[g]
            out[g] = dense[:, self.offsets[i]: self.offsets[i + 1]].reshape(B, n, H, W)
        return out

    def key(self) -> tuple:
        return tuple(tuple(self.shapes[g]) for g in self.geom.group_names)

    # ------------------------------------------------------------------ device tables
    def tables(self, device: torch.device) -> dict[str, torch.Tensor]:
        k = str(device)
        if k in self._dev_cache:
            return self._dev_cache[k]
        G = self.geom
        Hs = torch.tensor([self.shapes[g][0] for g in G.group_names], device=device)
        Ws = torch.tensor([self.shapes[g][1] for g in G.group_names], device=device)
        H0 = torch.tensor([G.group(g).shape[0] for g in G.group_names], device=device, dtype=torch.float32)
        W0 = torch.tensor([G.group(g).shape[1] for g in G.group_names], device=device, dtype=torch.float32)
        loff = torch.tensor([G.layer_offset[g] for g in G.group_names], device=device)
        t = dict(
            offsets=torch.tensor(self.offsets, device=device),
            H=Hs, W=Ws, H0=H0, W0=W0, loff=loff,
            depths=torch.tensor(G.depths_all, device=device, dtype=torch.float32),
        )
        self._dev_cache[k] = t
        return t

    def decompose(self, idx: torch.Tensor):
        """dense index -> (group, layer-in-group, i, j)."""
        t = self.tables(idx.device)
        g = torch.searchsorted(t["offsets"][1:], idx, right=True)
        local = idx - t["offsets"][g]
        H, W = t["H"][g], t["W"][g]
        l = local // (H * W)
        rem = local % (H * W)
        return g, l, rem // W, rem % W

    def compose(self, g, l, i, j) -> torch.Tensor:
        t = self.tables(i.device)
        H, W = t["H"][g], t["W"][g]
        return t["offsets"][g] + l * H * W + i * W + j

    def cell_info(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """coords (...,3) = (depth, u, v); level (...,2) = log2(res / native res); global layer id."""
        t = self.tables(idx.device)
        g, l, i, j = self.decompose(idx)
        H = t["H"][g].float()
        W = t["W"][g].float()
        pu, pv = self.geom.periodic
        fi, fj = i.float() + 0.5, j.float() + 0.5
        u = fi * TWO_PI / H if pu else -1.0 + fi * 2.0 / H
        v = fj * TWO_PI / W if pv else -1.0 + fj * 2.0 / W
        layer = t["loff"][g] + l
        depth = t["depths"][layer]
        coords = torch.stack([depth, u, v], dim=-1)
        level = torch.stack([torch.log2(H / t["H0"][g]), torch.log2(W / t["W0"][g])], dim=-1)
        return dict(coords=coords, level=level, layer=layer, group=g)

    def rank(self, device: torch.device) -> torch.Tensor:
        """Morton (Z-order) rank of every dense cell, grouped by (group, layer). Used so that
        packed token sequences are spatially local -> windowed attention = local attention."""
        k = "rank_" + str(device)
        if k in self._dev_cache:
            return self._dev_cache[k]
        keys = []
        for gi, gname in enumerate(self.geom.group_names):
            n = self.geom.group(gname).n_layers
            H, W = self.shapes[gname]
            ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            m = _morton(ii.ravel(), jj.ravel())
            for l in range(n):
                keys.append((np.int64(gi) << 56) | (np.int64(l) << 44) | m)
        keys = np.concatenate(keys)
        order = np.argsort(keys, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        r = torch.from_numpy(rank).to(device)
        self._dev_cache[k] = r
        return r


def _part1by1(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.int64) & 0x3FFFFF
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555
    return x


def _morton(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return _part1by1(i) | (_part1by1(j) << 1)


# ---------------------------------------------------------------------------------- factors / ladder
def _largest_divisor_leq(n: int, f: int) -> int:
    for d in range(min(n, f), 0, -1):
        if n % d == 0:
            return d
    return 1


def _smallest_prime_factor(n: int) -> int:
    for p in range(2, n + 1):
        if n % p == 0:
            return p
    return n


def lr_factors_for(geom: Geometry, f: int, overrides: dict | None = None) -> dict[str, tuple[int, int]]:
    """Per-group, per-axis downsampling factor for a nominal factor f.
    Rule: on super-resolved axes, the largest divisor of the axis length that is <= f
    (e.g. CaloChallenge r=9 with f=4 -> 3). Overrides: {str(f): {group: [fu, fv]}}."""
    out = {}
    ov = (overrides or {}).get(str(f), {}) if overrides else {}
    for g in geom.groups:
        if g.name in ov:
            out[g.name] = tuple(int(x) for x in ov[g.name])
            continue
        H, W = g.shape
        fu = _largest_divisor_leq(H, f) if g.sr_axes[0] else 1
        fv = _largest_divisor_leq(W, f) if g.sr_axes[1] else 1
        out[g.name] = (fu, fv)
    return out


@dataclass
class Ladder:
    """Coarse-to-fine scale ladder (VAR "next-scale prediction"). specs[0] = LR input,
    specs[native_index] = ground-truth resolution, later entries are above-native refinements."""
    specs: list[ScaleSpec]
    steps: list[dict[str, tuple[int, int]]]  # steps[k-1] = factor from specs[k-1] to specs[k]
    native_index: int

    @property
    def K(self) -> int:
        return len(self.steps)


def build_ladder(geom: Geometry, lr_factors: dict[str, tuple[int, int]], extra_steps: int = 0,
                 one_shot: bool = False) -> Ladder:
    """one_shot=True is the ablation without next-scale prediction (a single LR->native step)."""
    remaining = {g: list(lr_factors[g]) for g in geom.group_names}
    if one_shot:
        shapes0 = {g: (geom.group(g).shape[0] // lr_factors[g][0], geom.group(g).shape[1] // lr_factors[g][1])
                   for g in geom.group_names}
        specs = [geom.spec(shapes0), geom.native_spec()]
        steps = [{g: tuple(lr_factors[g]) for g in geom.group_names}]
        shapes = geom.native_shapes()
        for _ in range(extra_steps):
            step = {g: (2 if geom.group(g).sr_axes[0] else 1, 2 if geom.group(g).sr_axes[1] else 1)
                    for g in geom.group_names}
            shapes = {g: (shapes[g][0] * step[g][0], shapes[g][1] * step[g][1]) for g in geom.group_names}
            specs.append(geom.spec(shapes))
            steps.append(step)
        return Ladder(specs=specs, steps=steps, native_index=1)
    shapes = {g: (geom.group(g).shape[0] // lr_factors[g][0], geom.group(g).shape[1] // lr_factors[g][1])
              for g in geom.group_names}
    for g in geom.group_names:
        H, W = geom.group(g).shape
        assert H % lr_factors[g][0] == 0 and W % lr_factors[g][1] == 0, (g, lr_factors[g], (H, W))
    specs = [geom.spec(shapes)]
    steps: list[dict[str, tuple[int, int]]] = []
    while any(r > 1 for g in remaining for r in remaining[g]):
        step = {}
        for g in geom.group_names:
            s = []
            for a in range(2):
                R = remaining[g][a]
                f = 1 if R == 1 else (2 if R % 2 == 0 else _smallest_prime_factor(R))
                remaining[g][a] = R // f
                s.append(f)
            step[g] = tuple(s)
        shapes = {g: (shapes[g][0] * step[g][0], shapes[g][1] * step[g][1]) for g in geom.group_names}
        specs.append(geom.spec(shapes))
        steps.append(step)
    native_index = len(steps)
    for _ in range(extra_steps):
        step = {g: (2 if geom.group(g).sr_axes[0] else 1, 2 if geom.group(g).sr_axes[1] else 1)
                for g in geom.group_names}
        shapes = {g: (shapes[g][0] * step[g][0], shapes[g][1] * step[g][1]) for g in geom.group_names}
        specs.append(geom.spec(shapes))
        steps.append(step)
    return Ladder(specs=specs, steps=steps, native_index=native_index)


_STEP_CACHE: dict = {}


def step_tables(geom: Geometry, step: dict[str, tuple[int, int]], max_children: int, device):
    """Per-group child slot offsets for one ladder step: (G, max_children, 2) offsets and validity."""
    key = (id(geom), tuple(tuple(step[g]) for g in geom.group_names), max_children, str(device))
    if key not in _STEP_CACHE:
        _STEP_CACHE[key] = _step_tables(geom, step, max_children, device)
    return _STEP_CACHE[key]


def _step_tables(geom: Geometry, step: dict[str, tuple[int, int]], max_children: int, device):
    G = len(geom.group_names)
    off = torch.zeros(G, max_children, 2, dtype=torch.long)
    valid = torch.zeros(G, max_children, dtype=torch.bool)
    fac = torch.zeros(G, 2, dtype=torch.long)
    for gi, g in enumerate(geom.group_names):
        su, sv = step[g]
        if su * sv > max_children:
            raise ValueError(f"step factor {su}x{sv} exceeds model.max_children={max_children}")
        fac[gi] = torch.tensor([su, sv])
        s = 0
        for a in range(su):
            for b in range(sv):
                off[gi, s] = torch.tensor([a, b])
                valid[gi, s] = True
                s += 1
    return off.to(device), valid.to(device), fac.to(device)


def children_index(parent_idx: torch.Tensor, parent_spec: ScaleSpec, child_spec: ScaleSpec,
                   step: dict[str, tuple[int, int]], max_children: int):
    """dense idx at parent scale (...,) -> child dense idx (..., C) and slot validity (..., C)."""
    off, valid, fac = step_tables(parent_spec.geom, step, max_children, parent_idx.device)
    g, l, i, j = parent_spec.decompose(parent_idx)
    su, sv = fac[g][..., 0:1], fac[g][..., 1:2]
    ci = i.unsqueeze(-1) * su + off[g][..., 0]
    cj = j.unsqueeze(-1) * sv + off[g][..., 1]
    gg = g.unsqueeze(-1).expand_as(ci)
    ll = l.unsqueeze(-1).expand_as(ci)
    sval = valid[g]
    ci = torch.where(sval, ci, torch.zeros_like(ci))
    cj = torch.where(sval, cj, torch.zeros_like(cj))
    cidx = child_spec.compose(gg, ll, ci, cj)
    return cidx, sval, fac[g]


def parent_index(child_idx: torch.Tensor, child_spec: ScaleSpec, parent_spec: ScaleSpec,
                 step: dict[str, tuple[int, int]]) -> torch.Tensor:
    max_c = max(su * sv for su, sv in step.values())
    _, _, fac = step_tables(child_spec.geom, step, max_c, child_idx.device)
    g, l, i, j = child_spec.decompose(child_idx)
    return parent_spec.compose(g, l, i // fac[g][..., 0], j // fac[g][..., 1])


# ---------------------------------------------------------------------------------- presets
def make_geometry(cfg) -> Geometry:
    name = cfg.data.name
    es = float(cfg.data.get("energy_scale", 1.0))
    if name == "qg":
        H = int(cfg.data.get("pad_to", 128))
        return Geometry("qg", [LayerGroup("jet", ["tracks", "ecal", "hcal"], (H, H), [-1.0, 0.0, 1.0])],
                        energy_scale=es, extras=dict(halfwidth=float(cfg.data.get("image_halfwidth", 1.0))))
    if name == "calogan":
        # all three layers span the same transverse extent (480 mm); coordinates in [-1, 1].
        groups = [
            LayerGroup("layer_0", ["layer_0"], (3, 96), [-1.0], sr_axes=tuple(cfg.data.get("sr_axes_l0", [False, True]))),
            LayerGroup("layer_1", ["layer_1"], (12, 12), [0.0]),
            LayerGroup("layer_2", ["layer_2"], (12, 6), [1.0]),
        ]
        return Geometry("calogan", groups, energy_scale=es, extras=dict(extent_mm=480.0))
    if name == "calochallenge":
        nz, na, nr = 45, 16, 9
        depths = list(np.linspace(-1, 1, nz))
        g = LayerGroup("shower", [f"z{i}" for i in range(nz)], (na, nr), depths, periodic=(True, False))
        r_edges = cfg.data.get("r_edges", None) or list(np.linspace(0, 9, nr + 1))
        return Geometry("calochallenge", [g], energy_scale=es, extras=dict(r_edges=list(map(float, r_edges))))
    raise ValueError(f"unknown dataset {name}")
