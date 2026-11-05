"""Download the three datasets from their official hosts into data/real/ (the paths configs/data/*.yaml use).

    python scripts/download_data.py                      # all three (~7 GB)
    python scripts/download_data.py --dataset calogan    # calogan | calochallenge | qg (repeatable)
    python scripts/download_data.py --root $SCRATCH/calonsf/data/real

Sources:
  CaloGAN           Mendeley Data "Electromagnetic Calorimeter Shower Images", doi:10.17632/pvn3xc3wy5.1
                    -> calogan/{eplus,gamma,piplus}.hdf5            (3 x 406 MB, SHA-256 checked)
  CaloChallenge DS2 Zenodo record 6366271, doi:10.5281/zenodo.6366271
                    -> calochallenge_v2/dataset_2_{1,2}.hdf5        (2 x 1.36 GB, MD5 checked)
  Quark-Gluon       ML4SCI CMS Open Data jet images, public CERNBox share
                    -> qg/*.parquet (3 files, 3.7 GB, size checked), converted to qg/quark-gluon.h5

Downloads resume where they stopped (re-run the same command) and finished files are skipped after
their checksum is verified. Only the Python standard library is needed for downloading; the QG
conversion needs pyarrow and h5py (use --no-convert to skip it).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "calonsf-download/1.0"}
MENDELEY_DATASET, MENDELEY_VERSION = "pvn3xc3wy5", 1
ZENODO_RECORD = "6366271"
ZENODO_FILES = ["dataset_2_1.hdf5", "dataset_2_2.hdf5"]
CERNBOX_TOKEN = "EYgmOkI9BjwxNqy"
CERNBOX_DAV = f"https://cernbox.cern.ch/remote.php/dav/public-files/{CERNBOX_TOKEN}"


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


def download_calogan(root):
    out = os.path.join(root, "calogan")
    files = get_json(f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET}/files"
                     f"?folder_id=root&version={MENDELEY_VERSION}")
    by_name = {f["filename"]: f for f in files}
    for name in ("eplus.hdf5", "gamma.hdf5", "piplus.hdf5"):
        c = by_name[name]["content_details"]
        fetch(c["download_url"], os.path.join(out, name), c["size"], "sha256", c.get("sha256_hash"))


def download_calochallenge(root):
    out = os.path.join(root, "calochallenge_v2")
    meta = {f["key"]: f for f in get_json(f"https://zenodo.org/api/records/{ZENODO_RECORD}")["files"]}
    for name in ZENODO_FILES:
        f = meta[name]
        algo, digest = f["checksum"].split(":", 1)
        fetch(f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1", os.path.join(out, name),
              f["size"], algo, digest)


def list_cernbox():
    body = ('<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop><d:resourcetype/>'
            '<d:getcontentlength/></d:prop></d:propfind>')
    req = urllib.request.Request(CERNBOX_DAV + "/", data=body.encode(), method="PROPFIND",
                                 headers={**UA, "Depth": "1", "Content-Type": "application/xml"})
    with urllib.request.urlopen(req, timeout=60) as r:
        tree = ET.fromstring(r.read())
    ns = {"d": "DAV:"}
    out = []
    for resp in tree.findall("d:response", ns):
        name = urllib.parse.unquote(resp.find("d:href", ns).text).rstrip("/").rsplit("/", 1)[-1]
        size = resp.find(".//d:getcontentlength", ns)
        if name.endswith(".parquet") and size is not None:
            out.append((name, int(size.text)))
    return sorted(out)


def download_qg(root, convert=True, keep_parquet=True):
    out = os.path.join(root, "qg")
    h5 = os.path.join(out, "quark-gluon.h5")
    files = list_cernbox()
    if not files:
        raise RuntimeError(f"no parquet files listed in the CERNBox share {CERNBOX_DAV}")
    paths = []
    for name, size in files:
        p = os.path.join(out, name)
        fetch(f"{CERNBOX_DAV}/{urllib.parse.quote(name)}", p, size)
        paths.append(p)
    if not convert:
        print(f"[skip] conversion; run: python scripts/convert_qg_parquet.py --out {h5} {' '.join(paths)}")
        return
    if os.path.exists(h5):
        print(f"[ok  ] {h5} already exists (delete it to convert again)")
    else:
        print(f"[conv] {len(paths)} parquet files -> {h5}")
        tmp = h5 + ".tmp"
        conv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "convert_qg_parquet.py")
        subprocess.run([sys.executable, conv, "--out", tmp, *paths], check=True)
        os.replace(tmp, h5)
        print(f"[done] {h5}")
    if not keep_parquet:
        for p in paths:
            os.remove(p)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", action="append", choices=["calogan", "calochallenge", "qg", "all"],
                   help="repeatable; default all")
    p.add_argument("--root", default="data/real", help="where calogan/, calochallenge_v2/, qg/ go")
    p.add_argument("--no-convert", action="store_true", help="QG: keep the parquet files, do not build the HDF5")
    p.add_argument("--delete-parquet", action="store_true", help="QG: remove the parquet files after converting")
    a = p.parse_args()
    which = set(a.dataset or ["all"])
    if "all" in which:
        which = {"calogan", "calochallenge", "qg"}
    print(f"downloading {sorted(which)} into {os.path.abspath(a.root)}")
    failed = []
    for name, fn in (("calogan", lambda: download_calogan(a.root)),
                     ("calochallenge", lambda: download_calochallenge(a.root)),
                     ("qg", lambda: download_qg(a.root, not a.no_convert, not a.delete_parquet))):
        if name not in which:
            continue
        print(f"\n==== {name}")
        try:
            fn()
        except Exception as e:  # keep going with the other datasets
            print(f"!! {name} failed: {e}")
            failed.append(name)
    if failed:
        sys.exit(f"\nfailed: {failed}. Re-run the same command to resume.")
    print("\nall requested datasets are in place. Check them with:\n"
          "  python scripts/check_data.py --config configs/experiments/<calogan|qg|calochallenge>.yaml")


if __name__ == "__main__":
    main()
