"""Stream the ML4SCI Quark-Gluon parquet files into one HDF5 file (random access for training).

    python scripts/convert_qg_parquet.py --out data/real/qg/quark-gluon.h5 file1.parquet file2.parquet ...
Columns expected: X_jets (3x125x125 nested lists, or 125x125x3), y, and optionally pt, m0.
"""
import argparse

import h5py
import numpy as np
import pyarrow.parquet as pq

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("files", nargs="+")
p.add_argument("--out", required=True)
p.add_argument("--batch", type=int, default=256)
a = p.parse_args()
with h5py.File(a.out, "w") as f:
    X = f.create_dataset("X_jets", (0, 125, 125, 3), maxshape=(None, 125, 125, 3), dtype="f4",
                         chunks=(1, 125, 125, 3), compression="gzip")
    cols = {k: f.create_dataset(k, (0,), maxshape=(None,), dtype="f4") for k in ("y", "pt", "m0")}
    n = 0
    for path in a.files:
        pf = pq.ParquetFile(path)
        for rb in pf.iter_batches(batch_size=a.batch):
            d = rb.to_pydict()
            x = np.stack([np.array([np.stack(c) for c in row]) if isinstance(row[0], list) else np.array(row)
                          for row in d["X_jets"]]).astype("f4")
            if x.shape[1] == 3:
                x = x.transpose(0, 2, 3, 1)
            assert x.shape[1:] == (125, 125, 3), x.shape
            m = len(x)
            X.resize(n + m, 0)
            X[n:] = x
            for k, ds in cols.items():
                ds.resize(n + m, 0)
                ds[n:] = np.asarray(d[k], dtype="f4") if k in d else -1
            n += m
            print(f"{n} jets", end="\r")
print(f"\nwrote {n} jets to {a.out}")
