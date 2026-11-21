"""Energy-conserving resampling of a polar (alpha, r) grid onto a Cartesian N x N pixel grid.

CaloChallenge Dataset 2 layers are polar grids (alpha x r), so image sizes such as 256 x 256 ... 4096 x 4096
mean a Cartesian x-y image with x = eta = r cos(alpha) and y = phi = r sin(alpha) (the challenge's own
coordinate convention). The image covers the square [-R, R]^2, R = the outer radius, so no voxel falls outside.

Each polar cell is covered with `samples` x `samples` points (uniform in area: uniform in alpha and in r^2) and
its energy is split equally over them; each point deposits its share into the pixel it lies in. Every share is
deposited, so the total energy is conserved exactly (up to float rounding) for any pixel size, and the
pixel-resolved result approximates the exact area overlap of cell and pixel. Sizes where a polar cell is
much larger than a pixel would need more samples."""
from __future__ import annotations

import math

import numpy as np
import torch

from .features import refine_r_edges


class PolarToCartesian:
    def __init__(self, n_alpha: int, n_r: int, r_edges, size: int, samples: int = 4, device="cpu", chunk: int = 64):
        self.A, self.R, self.size, self.M = n_alpha, n_r, int(size), int(samples)
        e = np.asarray(r_edges, np.float64)
        e = e if len(e) == n_r + 1 else refine_r_edges(e, n_r)
        self.half = float(e[-1])  # outer radius: the image spans [-half, half] in x and y
        self.pixel = 2 * self.half / self.size
        self.device = torch.device(device)
        self.dtype = torch.float32 if self.device.type == "mps" else torch.float64  # MPS has no float64
        M = self.M
        r0, r1 = torch.as_tensor(e[:-1]), torch.as_tensor(e[1:])
        frac = (torch.arange(M, dtype=torch.float64) + 0.5) / M
        rs = (r0[:, None] ** 2 + (r1 - r0)[:, None] * (r1 + r0)[:, None] * frac[None]).sqrt()  # (R, M) area-uniform
        idx = torch.empty(n_alpha * n_r, M * M, dtype=torch.int32)
        for a0 in range(0, n_alpha, chunk):
            a = torch.arange(a0, min(a0 + chunk, n_alpha), dtype=torch.float64)
            al = (a[:, None] + frac[None]) * 2 * math.pi / n_alpha  # (ac, M)
            x = torch.cos(al)[:, None, :, None] * rs[None, :, None, :]  # (ac, R, M, M)
            y = torch.sin(al)[:, None, :, None] * rs[None, :, None, :]
            ix = ((x + self.half) / self.pixel).floor().long().clamp(0, self.size - 1)
            iy = ((y + self.half) / self.pixel).floor().long().clamp(0, self.size - 1)
            idx[a0 * n_r:(a0 + len(a)) * n_r] = (iy * self.size + ix).reshape(-1, M * M).int()
        self.map = idx.to(self.device)

    @torch.no_grad()
    def __call__(self, energy: torch.Tensor) -> torch.Tensor:
        """energy (B, A, R) -> image (B, N, N), rows = y (phi), columns = x (eta); image.sum() == energy.sum()."""
        B = energy.shape[0]
        out = torch.zeros(B, self.size * self.size, dtype=self.dtype, device=self.device)
        w_ = 1.0 / (self.M * self.M)
        for b in range(B):
            flat = energy[b].to(self.device, self.dtype).reshape(-1)
            nz = flat.nonzero().squeeze(1)  # empty cells deposit nothing
            vals = (flat[nz] * w_).repeat_interleave(self.M * self.M)
            out[b].index_add_(0, self.map[nz].reshape(-1).long(), vals)
        return out.view(B, self.size, self.size)

    def pixel_centres(self) -> np.ndarray:
        return -self.half + (np.arange(self.size) + 0.5) * self.pixel


def image_centre_of_energy(img: torch.Tensor, pix: PolarToCartesian) -> np.ndarray:
    """(B, N, N) -> (B, 2) centre of energy (x = eta, y = phi) in mm."""
    c = torch.as_tensor(pix.pixel_centres(), dtype=torch.float64)
    im = img.detach().cpu().double()
    tot = im.sum((-1, -2)).clamp(min=1e-300)
    x = (im.sum(-2) * c).sum(-1) / tot  # columns
    y = (im.sum(-1) * c).sum(-1) / tot  # rows
    return torch.stack([x, y], -1).numpy()
