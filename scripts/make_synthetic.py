"""Write toy datasets in the real file schemas to data/synthetic (for laptop smoke runs).

    python scripts/make_synthetic.py --n-qg 3000 --n-calogan 2000 --n-cc 3000
"""
import argparse

from calonsf.data.synthetic import make_all

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", default="data/synthetic")
p.add_argument("--n-qg", type=int, default=3000)
p.add_argument("--n-calogan", type=int, default=2000)
p.add_argument("--n-cc", type=int, default=3000)
a = p.parse_args()
make_all(a.root, a.n_qg, a.n_calogan, a.n_cc)
print(f"wrote synthetic data to {a.root}")
