"""Download the November 2026 replay dataset and check every file against its SHA256SUMS.

    python -m clashai.data.download

Files land in ``<data>/datasets/IL_Replay_1115``. A file already present with the right hash is
not fetched again; a file whose hash does not match is deleted and the run fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from .. import paths

REPO = "A1A2CV3/IL_Replay_1115"
REVISION = "main"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
HEADERS = {"User-Agent": "clashai-dataset-download"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r:
        return r.read()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dataset_dir() -> Path:
    return paths.data_root() / "datasets" / "IL_Replay_1115"


def download(out: Path | None = None) -> Path:
    out = out or dataset_dir()
    out.mkdir(parents=True, exist_ok=True)
    files = [i["path"] for i in json.loads(_get(API)) if i["type"] == "file"]
    sums_text = _get(RESOLVE + "SHA256SUMS").decode()
    (out / "SHA256SUMS").write_text(sums_text, encoding="utf-8")
    sums = {}
    for line in sums_text.splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.lstrip("*").removeprefix("./")] = digest.lower()
    unverified = [f for f in files if f not in sums and f != "SHA256SUMS"]
    for i, name in enumerate(files, 1):
        if name == "SHA256SUMS":
            continue
        target = out / name
        if target.exists() and name in sums and _sha256(target) == sums[name]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        part.write_bytes(_get(RESOLVE + name))
        if name in sums and _sha256(part) != sums[name]:
            part.unlink()
            raise RuntimeError(f"{name}: hash does not match SHA256SUMS")
        part.replace(target)
        print(f"[{i}/{len(files)}] {name}", flush=True)
    print(f"{len(files)} files in {out}; verified against SHA256SUMS except: {unverified or 'none'}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the IL_Replay_1115 dataset")
    parser.add_argument("--out", type=Path, default=None)
    download(parser.parse_args(argv).out)


if __name__ == "__main__":
    main()
