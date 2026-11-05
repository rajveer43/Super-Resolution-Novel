"""Generic training loop: AdamW, warmup + cosine or warmup-constant-decay (V-JEPA 2), grad clipping,
CSV logging, best/last checkpoints and resume."""
from __future__ import annotations

import math
import os
import time
from typing import Callable

import torch

from ..utils.misc import CSVLogger, free_cache, get_logger, to_device

log = get_logger()


def lr_lambda(schedule: str, total: int, warmup: int, decay_frac: float = 0.2):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if schedule == "wcd":
            start_decay = int(total * (1 - decay_frac))
            if step < start_decay:
                return 1.0
            return max(0.0, 1 - (step - start_decay) / max(1, total - start_decay))
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return f


def fit(model: torch.nn.Module, loss_fn: Callable[[dict], dict], train_loader, val_loader, *, epochs: int,
        lr: float, weight_decay: float, device, out_dir: str, schedule: str = "cosine", warmup_frac: float = 0.05,
        grad_clip: float = 1.0, max_steps_per_epoch: int | None = None, param_groups=None, log_every: int = 20,
        on_step: Callable[[int, int], None] | None = None, on_epoch: Callable[[int], None] | None = None,
        tag: str = "train", save_extra: Callable[[], dict] | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    params = param_groups or [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader) if max_steps_per_epoch is None else min(len(train_loader), max_steps_per_epoch)
    total = max(1, epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(schedule, total, int(total * warmup_frac)))
    csv = CSVLogger(os.path.join(out_dir, f"{tag}_log.csv"))
    last = os.path.join(out_dir, "last.pt")
    start_epoch, step, best = 0, 0, float("inf")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        if ck.get("tag") == tag and ck.get("epoch", -1) + 1 < epochs:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_epoch, step, best = ck["epoch"] + 1, ck["step"], ck.get("best", best)
            log.info(f"[{tag}] resumed from epoch {start_epoch}")
    for epoch in range(start_epoch, epochs):
        if on_epoch:
            on_epoch(epoch)
        model.train()
        t0 = time.time()
        run = {}
        for i, batch in enumerate(train_loader):
            if i >= steps_per_epoch:
                break
            batch = to_device(batch, device)
            out = loss_fn(batch)
            loss = out["loss"]
            if not torch.isfinite(loss):
                log.warning(f"[{tag}] non-finite loss at step {step}, skipping batch")
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            free_cache(device)
            step += 1
            if on_step:
                on_step(step, total)
            for k, v in out.items():
                run[k] = run.get(k, 0.0) + float(v.detach() if isinstance(v, torch.Tensor) else v)
            if step % log_every == 0:
                msg = " ".join(f"{k}={run[k] / (i + 1):.4f}" for k in run)
                log.info(f"[{tag}] ep {epoch} it {i + 1}/{steps_per_epoch} lr {sched.get_last_lr()[0]:.2e} {msg}")
        tr = {f"train_{k}": v / max(1, min(steps_per_epoch, i + 1)) for k, v in run.items()}
        val = evaluate_loss(model, loss_fn, val_loader, device, max_steps=max(10, (max_steps_per_epoch or 10**9) // 4))
        row = {"epoch": epoch, "step": step, "time_s": time.time() - t0, **tr, **{f"val_{k}": v for k, v in val.items()}}
        csv.log(row)
        log.info(f"[{tag}] epoch {epoch} done in {row['time_s']:.0f}s  train_loss={tr.get('train_loss', float('nan')):.4f}"
                 f"  val_loss={val.get('loss', float('nan')):.4f}")
        extra = save_extra() if save_extra else {}
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
              "step": step, "best": best, "tag": tag, **extra}
        torch.save(ck, last)
        if val.get("loss", float("inf")) <= best:
            best = val["loss"]
            ck["best"] = best
            torch.save({"model": model.state_dict(), "epoch": epoch, **extra}, os.path.join(out_dir, "best.pt"))
    if not os.path.exists(os.path.join(out_dir, "best.pt")):
        torch.save({"model": model.state_dict(), "epoch": epochs - 1}, os.path.join(out_dir, "best.pt"))
    return {"best_val": best, "steps": step}


@torch.no_grad()
def evaluate_loss(model, loss_fn, loader, device, max_steps: int = 50) -> dict:
    if loader is None:
        return {}
    model.eval()
    acc, n = {}, 0
    for i, batch in enumerate(loader):
        if i >= max_steps:
            break
        out = loss_fn(to_device(batch, device))
        for k, v in out.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        free_cache(device)
        n += 1
    return {k: v / max(n, 1) for k, v in acc.items()}
