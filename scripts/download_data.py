"""Download CaloChallenge Dataset 2 from Zenodo into data/real/calochallenge_v2/ (the path configs/data/calochallenge.yaml uses).

    python scripts/download_data.py
    python scripts/download_data.py --root $SCRATCH/calonsf/data/real

Source: Zenodo record 6366271, doi:10.5281/zenodo.6366271 -> calochallenge_v2/dataset_2_{1,2}.hdf5
(2 x 1.36 GB, checksum verified). File 2 is the official evaluation set.

Downloads resume where they stopped (re-run the same command) and finished files are skipped after
their checksum is verified. Only the Python standard library is needed.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

UA = {"User-Agent": "calonsf-download/1.0"}
ZENODO_RECORD = "6366271"
ZENODO_FILES = ["dataset_2_1.hdf5", "dataset_2_2.hdf5"]


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def file_hash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verified(path, size=None, algo=None, digest=None):
    if not os.path.exists(path) or (size and os.path.getsize(path) != size):
        return False
    if algo and digest:
        print(f"  checking {algo} of {os.path.basename(path)} ...", flush=True)
        return file_hash(path, algo) == digest
    return True


def fetch(url, dest, size=None, algo=None, digest=None, retries=5):
    """Resumable download to dest (via dest.part), then size / checksum verification."""
    name = os.path.basename(dest)
    if verified(dest, size, algo, digest):
        print(f"[ok  ] {name} already downloaded and verified")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    if os.path.exists(dest):  # a wrong or truncated file: start over
        os.replace(dest, part)
        if not (size and os.path.getsize(part) < size):
            os.remove(part)
    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if size and have == size:
            break
        req = urllib.request.Request(url, headers={**UA, **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:  # server ignored the range: restart
                    have = 0
                total = size or (have + int(r.headers.get("Content-Length", 0)))
                print(f"[get ] {name}  {total / 1e9:.2f} GB" + (f" (resuming at {have / 1e9:.2f} GB)" if have else ""))
                t0, done, last = time.time(), have, 0.0
                with open(part, "ab" if have else "wb") as f:
                    while chunk := r.read(1 << 22):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 10:
                            rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                            print(f"       {done / 1e9:.2f} / {total / 1e9:.2f} GB  ({rate:.1f} MB/s)", flush=True)
                            last = time.time()
            break
        except Exception as e:  # network hiccup: retry from where it stopped
            print(f"  attempt {attempt}/{retries} failed: {e}", flush=True)
            if attempt == retries:
                raise
            time.sleep(10 * attempt)
    os.replace(part, dest)
    if not verified(dest, size, algo, digest):
        raise RuntimeError(f"{dest}: size or checksum does not match the source. Delete it and re-run.")
    print(f"[done] {name}")


def download_calochallenge(root):
    out = os.path.join(root, "calochallenge_v2")
    meta = {f["key"]: f for f in get_json(f"https://zenodo.org/api/records/{ZENODO_RECORD}")["files"]}
    for name in ZENODO_FILES:
        f = meta[name]
        algo, digest = f["checksum"].split(":", 1)
        fetch(f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1", os.path.join(out, name),
              f["size"], algo, digest)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/real", help="where calochallenge_v2/ goes")
    a = p.parse_args()
    print(f"downloading CaloChallenge Dataset 2 into {os.path.abspath(a.root)}")
    try:
        download_calochallenge(a.root)
    except Exception as e:
        sys.exit(f"!! download failed: {e}. Re-run the same command to resume.")
    print("\ndone. Check the data with:\n  python scripts/check_data.py --config configs/experiments/calochallenge.yaml")


if __name__ == "__main__":
    main()
