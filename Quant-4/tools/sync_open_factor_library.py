"""Best-effort sync of the open-source factor library references from GitHub.

Downloads the upstream definition files that ``Main.factor_library`` cites
(Microsoft qlib Alpha158 and the WorldQuant 101 Alphas paper index) into
``Quant-4/Data_Cache/factor_references/`` so the bundled subset can be audited
against the public sources. The sync is best-effort: on any network failure
the bundled registry in ``Main.factor_library.FACTOR_SPECS`` remains the
fallback and the script exits non-zero with a message (never mutates the
engine).

Usage:
    python tools/sync_open_factor_library.py [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "Data_Cache" / "factor_references"

# pinned, stable references (raw.githubusercontent URLs)
SOURCES = {
    "qlib_alpha158.py": "https://raw.githubusercontent.com/microsoft/qlib/main/qlib/contrib/data/handler.py",
    "wq101_arxiv_index.txt": "https://arxiv.org/abs/1601.00991",
    "factor_library_registry.json": None,  # local registry mirror
}


def _registry_mirror() -> dict:
    sys.path.insert(0, str(PROJECT_ROOT))
    from Main.factor_library import FACTOR_SPECS

    return {"factors": FACTOR_SPECS, "note": "bundled registry (fallback when network is unavailable)"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync open-source factor references from GitHub (best-effort)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--offline", action="store_true", help="write only the local registry mirror")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, url in SOURCES.items():
        target = args.out / name
        if url is None or args.offline:
            payload = json.dumps(_registry_mirror(), ensure_ascii=False, indent=2)
            target.write_text(payload, encoding="utf-8")
            print(f"[local] {name} <- bundled registry ({len(payload)} bytes)")
            continue
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            target.write_bytes(resp.content)
            digest = hashlib.sha256(resp.content).hexdigest()[:12]
            print(f"[ok] {name} <- {url} ({len(resp.content)} bytes, sha256:{digest})")
        except Exception as exc:
            ok = False
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
    print(f"\nFactor references: {args.out}")
    print("Bundled registry stays authoritative; upstream files are audit material only.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
