"""Write a toy CaloChallenge Dataset 2 in the real file schema to data/synthetic (for laptop smoke runs).

    python scripts/make_synthetic.py --n-cc 3000
"""
import argparse

from calonsf.data.synthetic import make_all

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", default="data/synthetic")
p.add_argument("--n-cc", type=int, default=3000)
a = p.parse_args()
make_all(a.root, a.n_cc)
print(f"wrote synthetic data to {a.root}")
