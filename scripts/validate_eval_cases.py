#!/usr/bin/env python3
"""Validate retrieval annotations and optionally regenerate their frozen manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from validate.eval_cases import build_manifest, load_and_validate, manifest_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    cases, edition, norms_path, errors = load_and_validate(root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"Validation failed: {len(errors)} error(s)")
        return 1

    manifest = build_manifest(cases, edition, norms_path)
    destination = root / "evals/retrieval_manifest.json"
    rendered = manifest_json(manifest)
    if args.write_manifest:
        destination.write_text(rendered, encoding="utf-8")
        print(f"Wrote {destination.relative_to(root)}")
    elif not destination.is_file():
        print(f"ERROR: frozen manifest is missing: {destination.relative_to(root)}")
        return 1
    elif destination.read_text(encoding="utf-8") != rendered:
        print("ERROR: frozen manifest is stale; run with --write-manifest after reviewing changes")
        return 1
    counts = manifest["counts"]
    print(
        f"Validated {counts['all']} cases: {counts['verified']} verified "
        f"({counts['dev']} dev / {counts['eval']} eval), "
        f"{counts['draft']} draft, {counts['excluded']} excluded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
