#!/usr/bin/env python3
"""Download MASSIVE 1.1 and extract the locales used by the benchmark."""
from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)


def main() -> None:
    config = yaml.safe_load((ROOT / "benchmark.yaml").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["all", *config["languages"]], default="all")
    args = parser.parse_args()
    codes = list(config["languages"]) if args.language == "all" else [args.language]

    archive = ROOT / "data" / "cache" / "amazon-massive-dataset-1.1.tar.gz"
    download(config["dataset_url"], archive)
    with tarfile.open(archive, "r:gz") as tar:
        members = {Path(member.name).name: member for member in tar.getmembers()}
        for code in codes:
            locale = config["languages"][code]["locale"]
            member = members.get(f"{locale}.jsonl")
            if member is None:
                raise SystemExit(f"locale {locale} is missing from the MASSIVE archive")
            source = tar.extractfile(member)
            if source is None:
                raise SystemExit(f"could not read {member.name}")
            rows = {"train": [], "dev": [], "test": []}
            for raw_line in source:
                row = json.loads(raw_line.decode("utf-8"))
                kept = {key: row[key] for key in ("id", "locale", "utt", "annot_utt")}
                rows[row["partition"]].append(kept)
            out_dir = ROOT / "data" / "raw" / code
            out_dir.mkdir(parents=True, exist_ok=True)
            for split, split_rows in rows.items():
                path = out_dir / f"{split}.jsonl"
                with path.open("w", encoding="utf-8") as stream:
                    for row in split_rows:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{code}/{split}: {len(split_rows)} utterances -> {path}")


if __name__ == "__main__":
    main()
