"""
Source registry check / downloader for data/raw/.

The registry lives in data/sources.json (path, url, whether the URL is a direct
download, licence, retrieval date, sha256).

    python -m src.ingest.download_sources              # report status (default)
    python -m src.ingest.download_sources --download   # fetch missing files with direct URLs
    python -m src.ingest.download_sources --record-hashes  # write sha256 of present files

Exit code is non-zero if any file is missing or a recorded hash does not match,
so `make data` fails loudly instead of building an index from a partial corpus.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from src.config import DATA_RAW_DIR, PROJECT_ROOT

REGISTRY_PATH = PROJECT_ROOT / "data" / "sources.json"


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_source(entry: dict, raw_dir: Path = DATA_RAW_DIR) -> str:
    """Return one of: ok, ok_unhashed, missing, hash_mismatch."""
    path = raw_dir / entry["path"]
    if not path.exists():
        return "missing"
    if not entry.get("sha256"):
        return "ok_unhashed"
    return "ok" if sha256_of(path) == entry["sha256"] else "hash_mismatch"


def download(entry: dict, raw_dir: Path = DATA_RAW_DIR, timeout: int = 60) -> bool:
    """Download one direct-URL source; returns True on success."""
    if not entry.get("url_is_direct") or not entry.get("url"):
        return False
    dest = raw_dir / entry["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(entry["url"], timeout=timeout) as resp, open(tmp, "wb") as f:
        f.write(resp.read())
    tmp.replace(dest)
    return True


def main(argv=None, registry_path: Path = REGISTRY_PATH, raw_dir: Path = DATA_RAW_DIR) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    p.add_argument("--record-hashes", action="store_true")
    args = p.parse_args(argv)

    registry = load_registry(registry_path)
    problems = 0
    for entry in registry["sources"]:
        status = check_source(entry, raw_dir)
        if status == "missing" and args.download:
            try:
                if download(entry, raw_dir):
                    status = check_source(entry, raw_dir)
            except OSError as e:
                print(f"  download failed for {entry['path']}: {e}")
        if args.record_hashes and status in ("ok_unhashed", "hash_mismatch"):
            entry["sha256"] = sha256_of(raw_dir / entry["path"])
            status = "ok"

        hint = ""
        if status == "missing":
            hint = f"  ← fetch manually from {entry['url']}" if entry.get("url") else ""
            problems += 1
        elif status == "hash_mismatch":
            problems += 1
        print(f"  [{status:13s}] {entry['path']}{hint}")

    if args.record_hashes:
        registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
        print(f"  Recorded hashes in {registry_path}")

    print(f"  {problems} problem(s) across {len(registry['sources'])} sources")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
