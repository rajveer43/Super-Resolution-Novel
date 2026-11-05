"""CaloNSF: Next-Scale Flow transformer for resolution-invariant calorimeter super-resolution.

Pieces and where they come from:
  * cell-set tokens, upsampling-by-repetition conditioning, conditional flow matching, ensemble
    sampling ............................................................ Graph Denoising (Kakati et al.)
  * coarse-to-fine next-scale prediction, one transformer shared across scales, a scale only sees
    coarser scales (block-causal) ....................................... VAR (Tian et al.)
  * JEPA-pretrained encoder, continuous axial RoPE, progressive resolution . V-JEPA 2 (Assran et al.)
  * conditional generation from a preprocessed LR input, denoising pipeline DiffLense (Reddy et al.)

New in this work:
  * Energy-conserving split: at every step each non-empty parent cell predicts how its energy is split
    over its children (softmax over child logits x occupancy mask). Children always sum to the parent,
    so sum-pooling the output at any scale gives back the coarser output exactly.
  * Flow matching on the child-fraction logits (not raw energies) -> scale-free targets.
  * One set of weights for any ladder -> 2x, 4x, 8x ... and above-native refinement (128 -> 2048).
  * Frozen-JEPA "physics perceptual" loss.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.geometry import Geometry, ScaleSpec, build_ladder, children_index, parent_index
from ..data.ops import sum_pool
from ..data.tokens import Cells, cells_from_dense, cells_from_idx, scatter_to_dense
from .blocks import Block, TimeEmbed
from .encoder import CellEmbed, CellEncoder, Conditioner, make_rope


@dataclass
class StepState:
    idx: torch.Tensor
    valid: torch.Tensor
    h: torch.Tensor
    spec: ScaleSpec
    step: dict


class FlowHead(nn.Module):
    """Velocity field v(x_t, t | h) on the child-fraction logits of each parent cell."""

    def __init__(self, d: int, C: int, mixer_layers: int, heads: int, rope, window):
        super().__init__()
        self.t_emb = TimeEmbed(d)
        self.inp = nn.Linear(2 * C, d)
        self.window = window
        self.mix = nn.ModuleList([Block(d, heads, rope, cond_dim=d) for _ in range(mixer_layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, C))

    def forward(self, h, x_t, sval, t, coords, valid, cond):
        z = h + self.inp(torch.cat([x_t * sval, sval.float()], dim=-1))
        c = self.t_emb(t) + cond
        if len(self.mix) == 0:
            z = z + c.unsqueeze(1)
        for blk in self.mix:
            z = blk(z, coords, valid, cond=c, window=self.window)
        return self.out(z) * sval


def profile_w1(dense_a: torch.Tensor, dense_b: torch.Tensor, spec: ScaleSpec) -> torch.Tensor:
    """Mean 1D Wasserstein distance between per-layer marginal profiles along u and v."""
    ga, gb = spec.to_grids(dense_a), spec.to_grids(dense_b)
    losses = []
    for g in ga:
        A, Bt = ga[g], gb[g]
        ta, tb = A.sum((-1, -2)), Bt.sum((-1, -2))
        valid = (tb > 0).float()
        for ax in (-1, -2):
            if A.shape[ax] == 1 or A.shape[-1 if ax == -2 else -2] == 0:
                continue
            pa = A.sum(ax) / ta.clamp(min=1e-12).unsqueeze(-1)
            pb = Bt.sum(ax) / tb.clamp(min=1e-12).unsqueeze(-1)
            if pa.shape[-1] < 2:
                continue
            w = (pa.cumsum(-1) - pb.cumsum(-1)).abs().mean(-1)
            losses.append((w * valid).sum() / valid.sum().clamp(min=1))
    return torch.stack(losses).mean() if losses else dense_a.new_zeros(())


class CaloNSF(nn.Module):
    def __init__(self, cfg, geom: Geometry):
        super().__init__()
        m = cfg.model
        self.cfg, self.geom = cfg, geom
        d = int(m.d_model)
        self.d = d
        self.C = int(m.get("max_children", 16))
        self.window = m.get("window", 1024)
        self.es = geom.energy_scale
        self.encoder = CellEncoder(m, geom)
        self.conditioner = Conditioner(d, m.get("conditioning", {}))
        rope = make_rope(m, geom)
        self.p_embed = CellEmbed(d, geom.n_layers_total, self.es, extra_in=3,
                                 abs_level=bool(m.get("abs_level_feature", True)), absolute_pos=rope is None)
        self.inherit = nn.Linear(d, d)
        self.root = nn.Parameter(torch.zeros(d))
        self.dec = nn.ModuleList([Block(d, int(m.heads), rope, float(m.get("mlp_ratio", 4)), cross=True, cond_dim=d,
                                        dropout=float(m.get("dropout", 0.0))) for _ in range(int(m.dec_layers))])
        self.dec_norm = nn.LayerNorm(d)
        self.occ_head = nn.Linear(d, self.C)
        self.gen_type = m.get("generator", "flow")
        fl = m.get("flow", {})
        self.y_std = float(fl.get("y_std", 3.0))
        self.eps = float(fl.get("eps", 1e-4))
        self.sigma_min = float(fl.get("sigma_min", 1e-4))
        if self.gen_type == "flow":
            self.flow = FlowHead(d, self.C, int(fl.get("mixer_layers", 1)), int(m.heads), rope, self.window)
        else:
            self.reg = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.C))
        dn = m.get("denoise_head", "auto")
        if dn == "auto":
            dn = bool(cfg.data.get("noise", {}).get("enabled", False) or
                      cfg.data.get("preprocess", {}).get("enabled", False))
        self.denoise = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)) if dn else None
        self.conservation = bool(m.get("conservation", True))
        self.use_occ = bool(m.get("use_occupancy", True))
        self.occ_thr = float(m.get("occ_threshold", 0.5))
        self.one_shot = m.get("ladder", "next_scale") == "one_shot"
        self.percep = None
        self.percep_cond = None

    # ------------------------------------------------------------------ setup helpers
    def load_pretrained_encoder(self, jepa_state: dict) -> tuple[list, list]:
        enc = {k[len("encoder."):]: v for k, v in jepa_state.items() if k.startswith("encoder.")}
        con = {k[len("conditioner."):]: v for k, v in jepa_state.items() if k.startswith("conditioner.")}
        r1 = self.encoder.load_state_dict(enc, strict=False)
        self.conditioner.load_state_dict(con, strict=False)
        return r1.missing_keys, r1.unexpected_keys

    def set_perceptual_encoder(self) -> None:
        """Frozen copy of the (JEPA-pretrained) encoder used by the physics perceptual loss."""
        self.percep = copy.deepcopy(self.encoder)
        self.percep_cond = copy.deepcopy(self.conditioner)
        for p in list(self.percep.parameters()) + list(self.percep_cond.parameters()):
            p.requires_grad_(False)

    def ladder_for(self, factors: dict, extra_steps: int = 0):
        return build_ladder(self.geom, factors, extra_steps, one_shot=self.one_shot)

    # ------------------------------------------------------------------ building blocks
    def _encode(self, lr: torch.Tensor, spec: ScaleSpec, cond: torch.Tensor, train: bool):
        m = self.cfg.model
        cells = cells_from_dense(lr, spec, zero_frac=float(m.get("zero_token_frac", 0.0)) if train else 0.0,
                                 max_tokens=int(m.get("max_enc_tokens", 2048) if train else
                                                m.get("max_enc_tokens_eval", 8192)))
        total = lr.sum(1)
        enc = self.encoder(cells, total, cond)
        return cells, enc

    def _init_parents(self, cells: Cells, enc: torch.Tensor, lr: torch.Tensor, spec: ScaleSpec):
        if self.denoise is None:
            return lr, None
        pred = self.denoise(enc).squeeze(-1)
        thr = float(self.cfg.model.get("denoise_threshold", 0.0))
        e = self.es * torch.expm1(F.relu(pred))
        e = torch.where(pred > thr, e, torch.zeros_like(e))
        return scatter_to_dense(cells.idx, e, cells.valid, spec.T), pred

    def _prep(self, dense_prev, sp: ScaleSpec, sk: ScaleSpec, step: dict, state: StepState | None,
              max_parents: int | None):
        cells = cells_from_dense(dense_prev, sp, threshold=0.0, max_tokens=max_parents)
        cidx, sval, fac = children_index(cells.idx, sp, sk, step, self.C)
        sval = sval & cells.valid.unsqueeze(-1)
        nc = sval.sum(-1).float()
        extra = torch.stack([torch.log2(fac[..., 0].float()), torch.log2(fac[..., 1].float()), nc / self.C], -1)
        B, P = cells.idx.shape
        if state is None:
            inh = torch.zeros(B, P, self.d, device=dense_prev.device)
            has = torch.zeros(B, P, dtype=torch.bool, device=dense_prev.device)
        else:
            pidx = parent_index(cells.idx, sp, state.spec, state.step)
            T = state.spec.T
            lookup = torch.full((B, T + 1), -1, dtype=torch.long, device=dense_prev.device)
            rows = torch.arange(state.idx.shape[1], device=dense_prev.device).expand_as(state.idx)
            tgt = torch.where(state.valid, state.idx, torch.full_like(state.idx, T))
            lookup.scatter_(1, tgt, torch.where(state.valid, rows, torch.full_like(rows, -1)))
            r = lookup.gather(1, pidx)
            has = (r >= 0) & cells.valid
            inh = state.h.gather(1, r.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.d)) * has.unsqueeze(-1)
        return cells, cidx, sval, extra, inh, has

    def _decode(self, cells, extra, inh, has, total, enc, enc_cells, cond):
        x = self.p_embed(cells, total, extra) + self.inherit(inh) + (~has).unsqueeze(-1).float() * self.root
        x = x * cells.valid.unsqueeze(-1)
        for blk in self.dec:
            x = blk(x, cells.coords, cells.valid, ctx=enc, ctx_coords=enc_cells.coords, ctx_mask=enc_cells.valid,
                    cond=cond, window=self.window)
        return self.dec_norm(x) * cells.valid.unsqueeze(-1)

    def _fractions(self, x: torch.Tensor, sval: torch.Tensor, occ_prob: torch.Tensor | None = None):
        """Child logits (scaled) -> energy fractions. With conservation the fractions sum to 1."""
        y = x * self.y_std
        first = torch.zeros_like(sval)
        first[..., 0] = True
        sv = sval | (~sval.any(-1, keepdim=True) & first)
        if self.conservation:
            frac = torch.softmax(y.masked_fill(~sv, float("-inf")), dim=-1)
        else:
            frac = torch.exp(y.clamp(max=10)) * sv
        if occ_prob is not None and self.use_occ:
            keep = (occ_prob > self.occ_thr) & sv
            am = F.one_hot(frac.argmax(-1), frac.shape[-1]).bool()
            keep = keep | (~keep.any(-1, keepdim=True) & am)
            frac = frac * keep
            if self.conservation:
                frac = frac / frac.sum(-1, keepdim=True).clamp(min=1e-12)
        return frac * sval

    def _targets(self, cidx, sval, dense_k):
        B = cidx.shape[0]
        Ec = torch.gather(dense_k, 1, cidx.reshape(B, -1)).view_as(cidx) * sval
        Ep = Ec.sum(-1, keepdim=True)
        frac = Ec / Ep.clamp(min=1e-20)
        y = torch.log(frac + self.eps)
        if self.conservation:
            y = y - (y * sval).sum(-1, keepdim=True) / sval.sum(-1, keepdim=True).clamp(min=1)
        y = torch.where(sval, y, torch.zeros_like(y))
        return Ec, Ep.squeeze(-1), frac, y / self.y_std

    def _sample_flow(self, h, sval, cells, cond, nfe: int, solver: str):
        B, P, C = sval.shape
        x = torch.randn(B, P, C, device=h.device) * sval

        def f(t, x):
            tt = torch.full((B,), float(t), device=h.device)
            return self.flow(h, x, sval, tt, cells.coords, cells.valid, cond)

        if solver == "dopri5":
            from torchdiffeq import odeint
            return odeint(f, x, torch.tensor([0.0, 1.0], device=h.device), method="dopri5",
                          rtol=1e-4, atol=1e-4)[-1]
        ts = torch.linspace(0, 1, nfe + 1).tolist()
        for t0, t1 in zip(ts[:-1], ts[1:]):
            dt = t1 - t0
            if solver == "euler":
                x = x + dt * f(t0, x)
            elif solver == "midpoint":
                x = x + dt * f(t0 + dt / 2, x + dt / 2 * f(t0, x))
            elif solver == "rk4":
                k1 = f(t0, x)
                k2 = f(t0 + dt / 2, x + dt / 2 * k1)
                k3 = f(t0 + dt / 2, x + dt / 2 * k2)
                k4 = f(t1, x + dt * k3)
                x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(solver)
        return x

    # ------------------------------------------------------------------ training
    def loss(self, batch: dict) -> dict:
        cfg, geom = self.cfg, self.geom
        L = cfg.loss
        ladder = self.ladder_for(batch["factors"])
        spec0 = ladder.specs[0]
        lr = spec0.to_dense(batch["lr"])
        B = lr.shape[0]
        cond = self.conditioner(batch, B)
        enc_cells, enc = self._encode(lr, spec0, cond, train=True)
        out = {}
        total_loss = lr.new_zeros(())
        # ground truth at every scale of the ladder (sum-pooled HR)
        gts = [None] * (ladder.K + 1)
        gts[ladder.K] = ladder.specs[ladder.K].to_dense(batch["hr"])
        for k in range(ladder.K, 0, -1):
            gk = ladder.specs[k].to_grids(gts[k])
            st = ladder.steps[k - 1]

            gts[k - 1] = ladder.specs[k - 1].to_dense({g: sum_pool(gk[g], *st[g]) for g in gk})
        if self.denoise is not None:
            _, pred = self._init_parents(enc_cells, enc, lr, spec0)
            lc = spec0.to_dense(batch["lr_clean"])
            tgt = torch.log1p(torch.gather(lc, 1, enc_cells.idx) / self.es)
            l_den = ((pred - tgt).abs() * enc_cells.valid).sum() / enc_cells.valid.sum().clamp(min=1)
            out["denoise"] = l_den
            total_loss = total_loss + float(L.get("denoise", 1.0)) * l_den
        total = gts[0].sum(1)
        state = None
        maxp = cfg.model.get("max_parents_train", None)
        acc = {"fm": 0.0, "occ": 0.0, "profile": 0.0, "jepa": 0.0}
        for k, step in enumerate(ladder.steps):
            sp, sk = ladder.specs[k], ladder.specs[k + 1]
            cells, cidx, sval, extra, inh, has = self._prep(gts[k], sp, sk, step, state, maxp)
            h = self._decode(cells, extra, inh, has, total, enc, enc_cells, cond)
            Ec, Ep, frac_t, x1 = self._targets(cidx, sval, gts[k + 1])
            svf = sval.float()
            nsl = svf.sum().clamp(min=1)
            occ_logit = self.occ_head(h)
            l_occ = (F.binary_cross_entropy_with_logits(occ_logit, (Ec > 0).float(), reduction="none") * svf).sum() / nsl
            if self.gen_type == "flow":
                t = torch.rand(B, device=lr.device)
                tt = t.view(B, 1, 1)
                x0 = torch.randn_like(x1) * svf
                xt = (1 - (1 - self.sigma_min) * tt) * x0 + tt * x1
                u = x1 - (1 - self.sigma_min) * x0
                v = self.flow(h, xt, sval, t, cells.coords, cells.valid, cond)
                l_main = (((v - u) ** 2) * svf).sum() / nsl
                x1_hat = xt + (1 - tt) * v
            else:
                x1_hat = self.reg(h)
                if self.conservation:
                    logp = torch.log_softmax((x1_hat * self.y_std).masked_fill(~sval, -1e4), -1)
                    l_main = -(frac_t * logp * svf).sum() / cells.valid.float().sum().clamp(min=1)
                else:
                    l_main = (((x1_hat - x1) ** 2) * svf).sum() / nsl
            frac_hat = self._fractions(x1_hat, sval)
            E_hat = Ep.unsqueeze(-1) * frac_hat
            lstep = l_main + float(L.get("occ", 1.0)) * l_occ
            acc["fm"] += float(l_main.detach())
            acc["occ"] += float(l_occ.detach())
            if float(L.get("profile", 0.0)) > 0:
                dh = scatter_to_dense(cidx, E_hat, sval, sk.T)
                dt_ = scatter_to_dense(cidx, Ec, sval, sk.T)
                l_prof = profile_w1(dh, dt_, sk)
                lstep = lstep + float(L.profile) * l_prof
                acc["profile"] += float(l_prof.detach())
            if self.percep is not None and float(L.get("jepa", 0.0)) > 0 and k == ladder.K - 1:
                l_j = self._perceptual(cidx, sval, cells.valid, E_hat, Ec, sk, total, batch, B)
                lstep = lstep + float(L.jepa) * l_j
                acc["jepa"] += float(l_j.detach())
            total_loss = total_loss + lstep / ladder.K
            state = StepState(cells.idx, cells.valid, h, sp, step)
        for kname, v in acc.items():
            out[kname] = v / ladder.K
        out["loss"] = total_loss
        out["n_steps"] = ladder.K
        return out

    def _perceptual(self, cidx, sval, pvalid, E_hat, Ec, sk, total, batch, B):
        jp = int(self.cfg.loss.get("jepa_max_parents", 256))
        pri = torch.rand(pvalid.shape, device=cidx.device) + (~pvalid).float() * 2
        sel = torch.argsort(pri, 1)[:, :jp]
        g3 = lambda x: torch.gather(x, 1, sel.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        idx = g3(cidx).reshape(B, -1)
        val = (g3(sval) & torch.gather(pvalid, 1, sel).unsqueeze(-1)).reshape(B, -1)
        eh = g3(E_hat).reshape(B, -1)
        et = g3(Ec).reshape(B, -1)
        cells = cells_from_idx(idx, val, None, sk, energy=et)
        c = self.percep_cond(batch, B)
        with torch.no_grad():
            ft = F.layer_norm(self.percep(cells, total, c), (self.d,))
        fp = F.layer_norm(self.percep(cells, total, c, energy=eh), (self.d,))
        return ((fp - ft).abs().mean(-1) * val).sum() / val.sum().clamp(min=1)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def generate(self, batch: dict, extra_steps: int = 0, ensemble: int = 1, nfe: int | None = None,
                 solver: str | None = None, return_scales: bool = False, keep_samples: bool = False) -> dict:
        fl = self.cfg.model.get("flow", {})
        nfe = int(nfe or fl.get("nfe", 16))
        solver = solver or fl.get("solver", "midpoint")
        ladder = self.ladder_for(batch["factors"], extra_steps)
        spec0 = ladder.specs[0]
        lr = spec0.to_dense(batch["lr"])
        B = lr.shape[0]
        E = int(ensemble)
        b2 = {k: (v.repeat_interleave(E, 0) if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == B
                  else v) for k, v in batch.items()}
        lr = lr.repeat_interleave(E, 0)
        cond = self.conditioner(b2, B * E)
        enc_cells, enc = self._encode(lr, spec0, cond, train=False)
        dense, _ = self._init_parents(enc_cells, enc, lr, spec0)
        total = dense.sum(1)
        scales = [dense]
        state = None
        for k, step in enumerate(ladder.steps):
            sp, sk = ladder.specs[k], ladder.specs[k + 1]
            cells, cidx, sval, extra, inh, has = self._prep(dense, sp, sk, step, state, None)
            h = self._decode(cells, extra, inh, has, total, enc, enc_cells, cond)
            occ = torch.sigmoid(self.occ_head(h)) if self.use_occ else None
            x = self._sample_flow(h, sval, cells, cond, nfe, solver) if self.gen_type == "flow" else self.reg(h)
            frac = self._fractions(x, sval, occ)
            Ec = cells.energy.unsqueeze(-1) * frac
            dense = scatter_to_dense(cidx, Ec, sval, sk.T)
            state = StepState(cells.idx, cells.valid, h, sp, step)
            if return_scales:
                scales.append(dense)

        def agg(dn):
            x = dn.view(B, E, -1)
            return x.mean(1), (x.std(1) if E > 1 else torch.zeros_like(x[:, 0]))

        spec_f = ladder.specs[-1]
        mean, std = agg(dense)
        out = {"grids": spec_f.to_grids(mean), "std": spec_f.to_grids(std), "ladder": ladder, "spec": spec_f,
               "lr_init": spec0.to_grids(agg(scales[0])[0])}
        if keep_samples:
            out["samples"] = {g: v.view(B, E, *v.shape[1:]) for g, v in spec_f.to_grids(dense).items()}
        if return_scales:
            out["scales"] = [ladder.specs[i].to_grids(agg(d)[0]) for i, d in enumerate(scales)]
        return out
