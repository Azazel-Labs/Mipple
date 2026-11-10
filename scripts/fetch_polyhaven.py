#!/usr/bin/env python3
"""Fetch a fixed CC0 training set from Poly Haven (https://polyhaven.com, all assets CC0).

Used to fit the localized-search ranker (src/rdo.rs) on images that are not in any benchmark:
Kodak and RAD's Oodle set stay evaluation-only. Every `--step`-th texture by name is taken, at
1k, as its diffuse and OpenGL normal map PNGs, into test-images/polyhaven/ (gitignored). The md5
from the API is checked; existing files are kept. `--offset 13 --out polyhaven-test` gives a
disjoint held-out set for evaluating on wide (1k) textures.

    python scripts/fetch_polyhaven.py [--step 27] [--offset 0] [--res 1k] [--out polyhaven]
"""
import argparse
import hashlib
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
API = "https://api.polyhaven.com"
UA = {"User-Agent": "mipple-fetch/1.0"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=27)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--res", default="1k")
    ap.add_argument("--out", default="polyhaven")
    a = ap.parse_args()
    out = ROOT / "test-images" / a.out
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(json.loads(get(f"{API}/assets?t=textures")))[a.offset :: a.step]
    for tid in ids:
        files = json.loads(get(f"{API}/files/{tid}"))
        for kind, tag in (("Diffuse", "diff"), ("nor_gl", "nor")):
            f = files.get(kind, {}).get(a.res, {}).get("png")
            if not f:
                continue
            dst = out / f"{tid}_{tag}.png"
            if dst.exists() and hashlib.md5(dst.read_bytes()).hexdigest() == f["md5"]:
                continue
            data = get(f["url"])
            if hashlib.md5(data).hexdigest() != f["md5"]:
                raise SystemExit(f"md5 mismatch: {f['url']}")
            dst.write_bytes(data)
            print(dst.name, len(data))


if __name__ == "__main__":
    main()
