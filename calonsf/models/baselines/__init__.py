from .base import SRBaseline, project_energy
from .cnn import EDSR, LIIF, RCAN, Bicubic, Nearest, SwinIR
from .generative import DDPM, VARVQ, GraphFM

REGISTRY = {c.name: c for c in [Nearest, Bicubic, EDSR, RCAN, SwinIR, LIIF, DDPM, GraphFM, VARVQ]}


def make_baseline(name: str, cfg, geom, factor: int) -> SRBaseline:
    if name not in REGISTRY:
        raise ValueError(f"unknown baseline {name}; choose from {list(REGISTRY)}")
    return REGISTRY[name](cfg, geom, factor)
