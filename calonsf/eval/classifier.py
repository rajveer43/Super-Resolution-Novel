"""Binary classifier two-sample test (CaloChallenge section 8.3).

A classifier is trained to tell the Geant4 reference showers from the generated ones; the figure of merit is
the AUC on a held-out evaluation set (0.5 = indistinguishable, 1 = perfectly separable). Three variants:

  dnn_low   DNN on the low-level features: voxel energies / E_inc and log10(E_inc)
            (2 hidden layers of 2048, leaky ReLU 0.01, sigmoid, Adam 2e-4, batch 1000, 50 epochs)
  dnn_high  the same DNN on the high-level features of section 8.1
  cnn_low   3D ResNet-18 on the (45, 16, 9) shower image, E_inc batch-normalised and concatenated before the
            last layer (AdamW 2.5e-5, 48 epochs)

Reference and generated set are split into train / test / evaluation (60 / 20 / 20 %) separately per class.
The same event indices are used for both classes: the generated showers are the super-resolution of the
reference events, so a paired event must not sit in training for one class and in evaluation for the other.
The epoch with the highest test accuracy is kept, the classifier is calibrated with isotonic regression on the
test set, and the AUC is measured on the evaluation set. Everything is repeated over `seeds` random
initialisations (10 in the challenge) and the AUCs are averaged."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------------------------------------ models


class DNN(nn.Module):
    def __init__(self, d_in: int, hidden: int = 2048):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.LeakyReLU(0.01), nn.Linear(hidden, hidden),
                                 nn.LeakyReLU(0.01), nn.Linear(hidden, 1))  # sigmoid lives in the loss

    def forward(self, x, e=None):
        return self.net(x).squeeze(-1)


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm3d(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm3d(cout)
        self.skip = None if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))

    def forward(self, x):
        h = self.b2(self.c2(torch.relu(self.b1(self.c1(x)))))
        return torch.relu(h + (x if self.skip is None else self.skip(x)))


class ResNet3D(nn.Module):
    """18-layer 3D ResNet: 7^3 stride-2 conv, 3^3 stride-2 max-pool, 4 stages of 2 basic blocks (32..128
    channels), global average pooling, concatenation with the batch-normalised E_inc, linear classifier."""

    def __init__(self, channels=(32, 64, 128, 128)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, channels[0], 7, 2, 3, bias=False), nn.BatchNorm3d(channels[0]),
                                  nn.ReLU(), nn.MaxPool3d(3, 2, 1))
        blocks, cin = [], channels[0]
        for i, c in enumerate(channels):
            blocks += [_Block(cin, c, 1 if i == 0 else 2), _Block(c, c, 1)]
            cin = c
        self.body = nn.Sequential(*blocks)
        self.e_norm = nn.BatchNorm1d(1)
        self.fc = nn.Linear(cin + 1, 1)

    def forward(self, x, e):
        h = self.body(self.stem(x)).mean((2, 3, 4))
        return self.fc(torch.cat([h, self.e_norm(e[:, None])], 1)).squeeze(-1)


# ------------------------------------------------------------------------------------------ inputs
class _Source:
    """One class of the two-sample test: raw showers (N, Z, A, R) + incident energies, or a feature matrix."""

    def __init__(self, shower: np.ndarray, e_inc: np.ndarray, hlf: np.ndarray | None = None):
        self.shower, self.e_inc, self.hlf = shower, np.asarray(e_inc, np.float32), hlf

    def batch(self, idx: np.ndarray, kind: str, stats=None) -> tuple[np.ndarray, np.ndarray]:
        idx = np.sort(idx)
        e = self.e_inc[idx]
        if kind == "dnn_high":
            mu, sd = stats
            return ((self.hlf[idx] - mu) / sd).astype(np.float32), np.log10(e)
        x = self.shower[idx].astype(np.float32)
        if kind == "dnn_low":
            x = x.reshape(len(idx), -1) / e[:, None]
            return np.concatenate([x, np.log10(e)[:, None]], 1), np.log10(e)
        return np.log1p(x / e[:, None, None, None] * 1e3)[:, None], np.log10(e)  # cnn_low


def _make_batch(srcs, labels, rows, kind, stats, device):
    """rows: indices into each source; the batch holds the same rows of both classes."""
    xs, es, ys = [], [], []
    for s, lab in zip(srcs, labels):
        x, e = s.batch(rows, kind, stats)
        xs.append(x)
        es.append(e)
        ys.append(np.full(len(rows), lab, np.float32))
    return (torch.as_tensor(np.concatenate(xs)).to(device), torch.as_tensor(np.concatenate(es)).to(device),
            torch.as_tensor(np.concatenate(ys)).to(device))


@torch.no_grad()
def _predict(model, srcs, rows, kind, stats, device, bs):
    model.eval()
    out, ys = [], []
    for s in range(0, len(rows), bs):
        x, e, y = _make_batch(srcs, (0.0, 1.0), rows[s:s + bs], kind, stats, device)
        out.append(torch.sigmoid(model(x, e)).float().cpu())
        ys.append(y.cpu())
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def run_binary(kind: str, ref: _Source, gen: _Source, seeds: int = 10, epochs: int = 50, lr: float = 2e-4,
               batch: int = 1000, device="cpu", split=(0.6, 0.2, 0.2), seed0: int = 0) -> dict:
    """Train `seeds` classifiers of type `kind` and return the mean / std of the evaluation-set AUC."""
    n = min(len(ref.e_inc), len(gen.e_inc))
    perm = np.random.default_rng(seed0).permutation(n)
    ntr, nte = int(split[0] * n), int(split[1] * n)
    tr, te, ev = perm[:ntr], perm[ntr:ntr + nte], perm[ntr + nte:]
    srcs = (ref, gen)
    stats = None
    if kind == "dnn_high":
        a = ref.hlf[tr]
        stats = (a.mean(0), np.maximum(a.std(0), 1e-6))
    aucs, accs = [], []
    for sd in range(seeds):
        torch.manual_seed(seed0 * 1000 + sd)
        rng = np.random.default_rng(seed0 * 1000 + sd)
        if kind == "cnn_low":
            model = ResNet3D().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            d_in = (ref.hlf.shape[1] if kind == "dnn_high" else ref.shower[0].size + 1)
            model = DNN(d_in).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
        best_acc, best_state = -1.0, None
        for _ in range(epochs):
            model.train()
            order = rng.permutation(tr)
            for s in range(0, len(order), max(batch // 2, 1)):  # `batch` events in total from both classes
                rows = order[s:s + max(batch // 2, 1)]
                if len(rows) < 2:
                    continue
                x, e, y = _make_batch(srcs, (0.0, 1.0), rows, kind, stats, device)
                loss = nn.functional.binary_cross_entropy_with_logits(model(x, e), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            p, y = _predict(model, srcs, te, kind, stats, device, batch)
            acc = float(((p > 0.5) == (y > 0.5)).mean())
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        model.load_state_dict(best_state)
        p_te, y_te = _predict(model, srcs, te, kind, stats, device, batch)
        p_ev, y_ev = _predict(model, srcs, ev, kind, stats, device, batch)
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_te, y_te)  # calibration on the test set
        aucs.append(float(roc_auc_score(y_ev, iso.predict(p_ev))))
        accs.append(best_acc)
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "aucs": aucs,
            "test_acc_mean": float(np.mean(accs)), "n_train": int(2 * len(tr)), "n_eval": int(2 * len(ev))}
