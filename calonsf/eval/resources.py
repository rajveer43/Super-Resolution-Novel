"""Generation timing (CaloChallenge section 8.6) and memory requirements (section 8.7).

Timing: wall-clock seconds per generated shower for several batch sizes, on the device the evaluation runs on
(the challenge used batch sizes 1, 100 and 10 000; Geant4 itself simulates one event at a time). The model is
already in memory, so model-loading time is not included, and the figure is the time to super-resolve a shower
from its LR input, not to generate it from noise. Memory: number of trainable parameters, and the number needed
at generation (the frozen perceptual-loss copy of the encoder is only used for training)."""
from __future__ import annotations

import torch

from ..data.degrade import make_loader
from ..utils.misc import sync, to_device

_TRAIN_ONLY = ("percep.", "percep_cond.")


def parameter_counts(model: torch.nn.Module) -> dict:
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gen = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(_TRAIN_ONLY))
    return {"trainable_in_training": int(train), "needed_for_generation": int(gen)}


@torch.no_grad()
def time_generation(generate, cfg, reader, geom, idx, factor: int, batch_sizes, n_events: int, device) -> dict:
    """generate(batch_on_device) -> output. Returns {batch_size: {"s_per_event", "n_events"}}."""
    out = {}
    for bs in batch_sizes:
        n = min(max(int(n_events), int(bs)), len(idx))
        bs_eff = min(int(bs), n)
        dl = make_loader(cfg, reader, idx[:n], geom, False, factors=[factor], batch_size=bs_eff, shuffle=False)
        batches = [to_device(b, device) for b in dl]
        generate(batches[0])  # warm-up (kernel compilation, allocator)
        sync(device)
        import time
        t0 = time.perf_counter()
        done = 0
        for b in batches:
            generate(b)
            done += len(b["e_inc"])
        sync(device)
        out[str(bs)] = {"s_per_event": (time.perf_counter() - t0) / max(done, 1), "n_events": int(done),
                        "batch_size_used": int(bs_eff)}
    return out
