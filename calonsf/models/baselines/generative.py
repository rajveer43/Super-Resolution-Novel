"""Generative baselines:
  * DDPM   - DiffLense-style conditional diffusion U-Net (cosine schedule, L1 on noise, LR condition).
  * GraphFM - Graph Denoising paper: fully connected transformer over HR cells (children of non-empty LR
              cells), upsampling-by-repetition condition, per-cell conditional flow matching, ensemble mean.
  * VARVQ  - VAR: multi-scale residual VQ-VAE + block-causal next-scale transformer, conditioned on LR.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.geometry import build_ladder, children_index
from ...data.ops import repeat_upsample, resample_energy


def area_resize(x, size):
    """Area-average resize via exact overlap matrices (works on MPS for any sizes)."""
    H, W = x.shape[-2:]
    return resample_energy(x, tuple(size)) * (size[0] * size[1]) / (H * W)
from ...data.tokens import cells_from_dense, scatter_to_dense
from ..blocks import Block, ContinuousRoPE, TimeEmbed, fourier
from .base import SRBaseline, from_t, pad_mult, to_t


# ============================================================================ DDPM (DiffLense)
def cosine_alphas(T: int) -> torch.Tensor:
    s = 0.008
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return ab[1:].clamp(1e-5, 1.0)


class _ResT(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n1 = nn.GroupNorm(8, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = F.silu(self.n1(self.c1(x))) + self.t(te)[:, :, None, None]
        h = F.silu(self.n2(self.c2(h)))
        return h + self.skip(x)


class _UNet(nn.Module):
    def __init__(self, n, c):
        super().__init__()
        self.te = TimeEmbed(4 * c)
        self.inp = nn.Conv2d(2 * n, c, 3, padding=1)
        self.d1 = _ResT(c, c, 4 * c)
        self.d2 = _ResT(c, 2 * c, 4 * c)
        self.mid = _ResT(2 * c, 2 * c, 4 * c)
        self.u2 = _ResT(4 * c, c, 4 * c)
        self.u1 = _ResT(2 * c, c, 4 * c)
        self.out = nn.Conv2d(c, n, 3, padding=1)

    def forward(self, x, cond, t):
        te = self.te(t)
        h0 = self.inp(torch.cat([x, cond], 1))
        h1 = self.d1(h0, te)
        down = h1.shape[-1] >= 4 and h1.shape[-2] >= 4
        h2 = self.d2(F.avg_pool2d(h1, 2) if down else h1, te)
        m = self.mid(h2, te)
        u = F.interpolate(m, size=h1.shape[-2:], mode="nearest") if down else m
        u = self.u2(torch.cat([u, h2 if not down else F.interpolate(h2, size=h1.shape[-2:])], 1), te)
        u = self.u1(torch.cat([u, h1], 1), te)
        return self.out(u)


class DDPM(SRBaseline):
    name = "ddpm"

    def __init__(self, cfg, geom, factor):
        super().__init__(cfg, geom, factor)
        c = int(cfg.baseline.channels)
        c = c - c % 8
        self.T = int(cfg.baseline.ddpm_steps)
        self.S = int(cfg.baseline.ddpm_sample_steps)
        self.register_buffer("ab", cosine_alphas(self.T))
        self.nets = nn.ModuleDict({g: _UNet(geom.group(g).n_layers, c) for g in geom.group_names})
        self.register_buffer("scale", torch.zeros(len(geom.group_names)))  # data -> [-1, 1]

    def _norm(self, gi, x):
        if self.scale[gi] == 0:
            self.scale[gi] = x.amax().clamp(min=1e-3)
        return x / self.scale[gi] * 2 - 1

    def _cond(self, g, lr_t):
        fu, fv = self.factors[g]
        return lr_t.repeat_interleave(fu, -2).repeat_interleave(fv, -1)

    def loss_group(self, g, lr_t, hr_t, batch):
        gi = self.geom.group_names.index(g)
        x0 = self._norm(gi, hr_t)
        cond = self._norm(gi, self._cond(g, lr_t))
        x0, hw = pad_mult(x0, 4)
        cond, _ = pad_mult(cond, 4)
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        ab = self.ab[t].view(B, 1, 1, 1)
        eps = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        pred = self.nets[g](xt, cond, t.float() / self.T)
        return (pred - eps).abs().mean()  # DiffLense: L1 between predicted and true noise

    def predict_group(self, g, lr_t, batch):
        gi = self.geom.group_names.index(g)
        cond = self._norm(gi, self._cond(g, lr_t))
        cond, hw = pad_mult(cond, 4)
        x = torch.randn_like(cond)
        ts = torch.linspace(self.T - 1, 0, self.S).long().tolist()
        for i, t in enumerate(ts):  # DDIM (eta=0) for speed
            ab = self.ab[t]
            ab_prev = self.ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=x.device)
            eps = self.nets[g](x, cond, torch.full((x.shape[0],), t / self.T, device=x.device))
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        x = x[..., :hw[0], :hw[1]]
        return (x + 1) / 2 * self.scale[gi]


# ============================================================================ Graph-FM (Kakati et al.)
class GraphFM(SRBaseline):
    name = "graphfm"

    def __init__(self, cfg, geom, factor):
        super().__init__(cfg, geom, factor)
        m = cfg.model
        d = int(m.d_model)
        self.d = d
        self.rope = ContinuousRoPE(d // int(m.heads), geom.periodic)
        self.inp = nn.Linear(2 + 2, d)
        self.layer = nn.Embedding(geom.n_layers_total, d)
        self.te = TimeEmbed(d)
        self.blocks = nn.ModuleList([Block(d, int(m.heads), self.rope, cond_dim=d) for _ in range(4)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        self.nfe = int(cfg.baseline.graphfm_nfe)
        self.window = m.get("window", 1024)
        self.ens = int(cfg.eval.get("ensemble", 4))
        self.lad = build_ladder(geom, self.factors, one_shot=True)
        self.C = max(su * sv for su, sv in self.lad.steps[0].values())
        self.maxp = int(m.get("max_parents_train", 3072)) // max(1, self.C // 4)

    def _graph(self, lr_dense, maxp=None):
        sp, sk = self.lad.specs[0], self.lad.specs[1]
        cells = cells_from_dense(lr_dense, sp, max_tokens=maxp)
        cidx, sval, _ = children_index(cells.idx, sp, sk, self.lad.steps[0], self.C)
        sval = sval & cells.valid.unsqueeze(-1)
        B = cidx.shape[0]
        idx = cidx.reshape(B, -1)
        val = sval.reshape(B, -1)
        info = sk.cell_info(idx)
        nchild = sval.sum(-1, keepdim=True).clamp(min=1).float()
        par = (cells.energy.unsqueeze(-1) / nchild).expand_as(cidx).reshape(B, -1)  # upsampling by repetition
        return idx, val, info, par, sk

    def _v(self, x, t, info, par, val):
        feat = torch.stack([to_t(par, self.es), x], -1)
        h = self.inp(torch.cat([feat, info["level"]], -1)) + self.layer(info["layer"])
        c = self.te(t)
        for b in self.blocks:
            h = b(h, info["coords"], val, cond=c, window=self.window)
        return self.out(h).squeeze(-1) * val

    def loss(self, batch):
        self._check(batch)
        sp = self.lad.specs[0]
        lr = sp.to_dense(batch["lr"])
        hr = self.lad.specs[1].to_dense(batch["hr"])
        idx, val, info, par, sk = self._graph(lr, self.maxp)
        x1 = to_t(torch.gather(hr, 1, idx), self.es) * val
        B = x1.shape[0]
        t = torch.rand(B, device=x1.device)
        x0 = torch.randn_like(x1)
        s = 1e-4
        xt = (1 - (1 - s) * t[:, None]) * x0 + t[:, None] * x1
        u = x1 - (1 - s) * x0
        v = self._v(xt, t, info, par, val)
        return {"loss": (((v - u) ** 2) * val).sum() / val.sum().clamp(min=1)}

    @torch.no_grad()
    def predict(self, batch):
        self._check(batch)
        sp = self.lad.specs[0]
        lr = sp.to_dense(batch["lr"]).repeat_interleave(self.ens, 0)
        idx, val, info, par, sk = self._graph(lr)
        x = torch.randn(idx.shape, device=lr.device)
        B = x.shape[0]
        for i in range(self.nfe):
            t = torch.full((B,), i / self.nfe, device=x.device)
            x = x + self._v(x, t, info, par, val) / self.nfe
        e = from_t(x, self.es) * val
        dense = scatter_to_dense(idx, e, val, sk.T)
        dense = dense.view(-1, self.ens, sk.T).mean(1)  # ensemble sampling (Graph Denoising Sec. III C)
        out = sk.to_grids(dense)
        if self.project:
            from .base import project_energy
            out = {g: project_energy(out[g], batch["lr"][g], *self.factors[g]) for g in out}
        return out


# ============================================================================ VAR-VQ (Tian et al.)
class _VQGroup(nn.Module):
    def __init__(self, n, c, cq, V, H, W):
        super().__init__()
        self.stride = 4 if min(H, W) >= 64 else 1
        down = [nn.Conv2d(n, c, 3, padding=1), nn.SiLU()]
        up = []
        if self.stride == 4:
            down += [nn.Conv2d(c, c, 4, 2, 1), nn.SiLU(), nn.Conv2d(c, c, 4, 2, 1), nn.SiLU()]
            up = [nn.ConvTranspose2d(c, c, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(c, c, 4, 2, 1), nn.SiLU()]
        self.enc = nn.Sequential(*down, nn.Conv2d(c, cq, 1))
        self.dec = nn.Sequential(nn.Conv2d(cq, c, 3, padding=1), nn.SiLU(), *up, nn.Conv2d(c, n, 3, padding=1))
        self.codebook = nn.Embedding(V, cq)
        nn.init.uniform_(self.codebook.weight, -1 / V, 1 / V)
        self.phi = nn.Conv2d(cq, cq, 3, padding=1)  # shared quant-residual conv (VAR phi_k)
        h, w = H // self.stride, W // self.stride
        self.hw = (h, w)
        self.scales = []
        k = 0
        while True:
            s = (min(2 ** k, h), min(2 ** k, w))
            self.scales.append(s)
            if s == (h, w):
                break
            k += 1

    def lookup(self, z):  # z (B, cq, h, w) -> idx (B, h*w)
        B, C, h, w = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(-1, C)
        d = (zf ** 2).sum(1, keepdim=True) - 2 * zf @ self.codebook.weight.t() + (self.codebook.weight ** 2).sum(1)
        return d.argmin(1).view(B, h * w)

    def embed_idx(self, idx, s):
        B = idx.shape[0]
        return self.codebook(idx).view(B, s[0], s[1], -1).permute(0, 3, 1, 2)

    def up(self, zq):
        u = F.interpolate(zq, size=self.hw, mode="bicubic", align_corners=False) if zq.shape[-2:] != self.hw else zq
        return u + 0.5 * self.phi(u)

    def quantize(self, f):
        """Algorithm 1 (multi-scale residual quantization). Returns f_hat and token maps."""
        rest, fhat, idxs = f, torch.zeros_like(f), []
        for s in self.scales:
            z = area_resize(rest, s) if s != self.hw else rest
            idx = self.lookup(z)
            idxs.append(idx)
            q = self.up(self.embed_idx(idx, s))
            fhat = fhat + q
            rest = rest - q
        return fhat, idxs


class _VARTransformer(nn.Module):
    def __init__(self, cq, d, V, n_lr, layers=4, heads=4):
        super().__init__()
        self.inp = nn.Linear(cq, d)
        self.lr_inp = nn.Linear(n_lr * 9, d)
        self.pos = nn.Linear(2 * 2 * 8, d)
        self.lvl = nn.Embedding(32, d)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
                                     for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, V))

    @staticmethod
    def grid_pos(s, device):
        yy = (torch.arange(s[0], device=device).float() + 0.5) / s[0] * 2 - 1
        xx = (torch.arange(s[1], device=device).float() + 0.5) / s[1] * 2 - 1
        return torch.stack(torch.meshgrid(yy, xx, indexing="ij"), -1).view(-1, 2)

    def forward(self, lr_t, inputs, scales):
        """lr_t (B, n, h, w); inputs: list of (B, cq, s) maps (input for scale k = up-interpolated f_hat_<k)."""
        B = lr_t.shape[0]
        lr_tok = F.unfold(lr_t, 3, padding=1).transpose(1, 2)  # (B, h*w, 9n)
        x = [self.lr_inp(lr_tok) + self.pos(fourier(self.grid_pos(lr_t.shape[-2:], lr_t.device), 8, 8.0))]
        lens = [x[0].shape[1]]
        for k, (inp, s) in enumerate(zip(inputs, scales)):
            t = self.inp(inp.flatten(2).transpose(1, 2)) + self.lvl.weight[k] + \
                self.pos(fourier(self.grid_pos(s, lr_t.device), 8, 8.0))
            x.append(t)
            lens.append(t.shape[1])
        x = torch.cat(x, 1)
        L = x.shape[1]
        blk = torch.cat([torch.full((n,), i, device=x.device) for i, n in enumerate(lens)])
        mask = blk[None, :] > blk[:, None]  # True = blocked (block-causal mask, VAR Fig. 4)
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.head(x[:, lens[0]:])


class VARVQ(SRBaseline):
    name = "varvq"

    def __init__(self, cfg, geom, factor):
        super().__init__(cfg, geom, factor)
        b = cfg.baseline
        c, cq, V = int(b.channels), int(b.vq_dim), int(b.vq_codebook)
        self.vq = nn.ModuleDict({g: _VQGroup(geom.group(g).n_layers, c, cq, V, *geom.group(g).shape)
                                 for g in geom.group_names})
        self.tf = nn.ModuleDict({g: _VARTransformer(cq, int(cfg.model.d_model), V, geom.group(g).n_layers)
                                 for g in geom.group_names})
        self.stage = 1
        self.stage1_frac = float(b.get("vq_stage1_frac", 0.5))

    def set_epoch(self, epoch: int, epochs: int):
        self.stage = 1 if epoch < max(1, int(round(epochs * self.stage1_frac))) else 2
        for p in self.vq.parameters():
            p.requires_grad_(self.stage == 1)

    def _inputs(self, vq, idxs):
        """Teacher-forcing inputs: for scale k, the cumulative f_hat of scales < k down-sampled to scale k."""
        cq = vq.codebook.weight.shape[1]
        B = idxs[0].shape[0]
        fhat = torch.zeros(B, cq, *vq.hw, device=idxs[0].device)
        ins = []
        for k, s in enumerate(vq.scales):
            ins.append(area_resize(fhat, s) if s != vq.hw else fhat)
            fhat = fhat + vq.up(vq.embed_idx(idxs[k], s))
        return ins

    def loss_group(self, g, lr_t, hr_t, batch):
        vq = self.vq[g]
        if self.stage == 1:
            f = vq.enc(hr_t)
            fhat, _ = vq.quantize(f)
            l_vq = F.mse_loss(fhat, f.detach()) + 0.25 * F.mse_loss(f, fhat.detach())
            fst = f + (fhat - f).detach()
            return (vq.dec(fst) - hr_t).abs().mean() + l_vq
        with torch.no_grad():
            _, idxs = vq.quantize(vq.enc(hr_t))
        logits = self.tf[g](lr_t, self._inputs(vq, idxs), vq.scales)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), torch.cat(idxs, 1).reshape(-1))

    def predict_group(self, g, lr_t, batch):
        vq = self.vq[g]
        cq = vq.codebook.weight.shape[1]
        B = lr_t.shape[0]
        fhat = torch.zeros(B, cq, *vq.hw, device=lr_t.device)
        idxs, ins = [], []
        for k, s in enumerate(vq.scales):
            ins.append(area_resize(fhat, s) if s != vq.hw else fhat)
            logits = self.tf[g](lr_t, ins, vq.scales[: k + 1])
            n = s[0] * s[1]
            idx = logits[:, -n:].argmax(-1)  # greedy decoding (SR wants fidelity, not diversity)
            idxs.append(idx)
            fhat = fhat + vq.up(vq.embed_idx(idx, s))
        return vq.dec(fhat)
