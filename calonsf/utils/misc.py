"""Device selection, seeding, logging, checkpoints."""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import time
from typing import Any

import numpy as np
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_logger(name: str = "calonsf", log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(fh)
    return logger


def run_dir(cfg) -> str:
    d = os.path.join(cfg.out_dir, cfg.exp_name)
    os.makedirs(d, exist_ok=True)
    return d


class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        self.fields: list[str] | None = None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, row: dict[str, Any]) -> None:
        row = {k: (float(v) if isinstance(v, (torch.Tensor, np.floating)) else v) for k, v in row.items()}
        new = self.fields is None or not os.path.exists(self.path)
        if self.fields is None:
            self.fields = list(row.keys())
        for k in row:
            if k not in self.fields:
                self.fields.append(k)
                new = True
        if new and os.path.exists(self.path):
            # rewrite with new header
            with open(self.path) as f:
                old = list(csv.DictReader(f))
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fields)
                w.writeheader()
                for r in old:
                    w.writerow(r)
                w.writerow(row)
            return
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            if new:
                w.writeheader()
            w.writerow(row)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def conv(x):
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        raise TypeError(type(x))

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def device_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        try:
            return torch.mps.driver_allocated_memory() / 2**20
        except Exception:
            return float("nan")
    return float("nan")


def free_cache(device: torch.device) -> None:
    """Release cached GPU blocks. On MPS, batches with a different number of tokens each time make the
    allocator keep every block size it has seen; without this, driver memory grew past 24 GB and the Mac
    swapped (training slowed from 1.6 to ~10 s/step). Clearing every step keeps it at ~0.2 GB."""
    if device.type == "mps":
        torch.mps.empty_cache()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
