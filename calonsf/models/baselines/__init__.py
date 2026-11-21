"""Non-trainable reference upsamplers: nearest-neighbour repetition and bicubic interpolation."""
from .interpolation import Bicubic, Nearest, SRBaseline, project_energy

REGISTRY = {c.name: c for c in (Nearest, Bicubic)}


def make_baseline(name: str, cfg, geom, factor: int) -> SRBaseline:
    if name not in REGISTRY:
        raise ValueError(f"unknown baseline {name}; choose from {list(REGISTRY)}")
    return REGISTRY[name](cfg, geom, factor)
