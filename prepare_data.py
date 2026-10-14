#!/usr/bin/env python3
"""Turn MASSIVE slot annotations into fixed, balanced word-level samples."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
EVAL_SEED = 1000
SLOT = re.compile(r"\[\s*([^:\]]+?)\s*:\s*([^\]]*?)\s*\]")


def add_words(rows: list[dict], text: str, label: str, slot_type: str | None) -> None:
    for word in text.split():
        rows.append({"word": word, "label": label, "slot_type": slot_type})


def parse_annotation(text: str) -> list[dict]:
    words: list[dict] = []
    end = 0
    for match in SLOT.finditer(text):
        add_words(words, text[end:match.start()], "O", None)
        add_words(words, match.group(2), "SLOT", match.group(1).strip())
        end = match.end()
    add_words(words, text[end:], "O", None)
    if "[" in " ".join(item["word"] for item in words) or "]" in " ".join(item["word"] for item in words):
        raise ValueError(f"could not parse slot annotation: {text}")
    return words


def read_examples(path: Path) -> list[dict]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        utterance = json.loads(line)
        annotated = parse_annotation(utterance["annot_utt"])
        words = [item["word"] for item in annotated]
        for word_id, item in enumerate(annotated):
            examples.append({
                "utterance_id": utterance["id"],
                "locale": utterance["locale"],
                "words": words,
                "word_id": word_id,
                **item,
            })
    return examples


def balanced(rows: list[dict], cap: int, seed: int) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    rng = np.random.default_rng(seed)
    selected = []
    for label in ("O", "SLOT"):
        group = by_label[label]
        if len(group) < cap:
            raise ValueError(f"only {len(group)} examples for {label}; need {cap}")
        selected.extend(group[i] for i in rng.choice(len(group), cap, replace=False))
    return sorted(selected, key=lambda row: (row["utterance_id"], row["word_id"]))


def parallel_pool(rows: list[dict], size: int, seed: int) -> list[dict]:
    """Select request IDs deterministically, giving every locale the same pool."""
    utterance_ids = sorted({row["utterance_id"] for row in rows})
    if len(utterance_ids) < size:
        raise ValueError(f"only {len(utterance_ids)} utterances; need {size}")
    rng = np.random.default_rng(seed)
    selected = {utterance_ids[i] for i in rng.choice(len(utterance_ids), size, replace=False)}
    return [row for row in rows if row["utterance_id"] in selected]


def write_split(language: str, name: str, rows: list[dict]) -> None:
    out = ROOT / "data" / "prepared" / language
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{language}/{name}: {len(rows)} words sha256={digest}")


def prepare(language: str, config: dict) -> None:
    caps = config["sample_cap_per_label"]
    pool_sizes = config["utterance_pool_size"]
    for split in ("train", "dev", "test"):
        rows = read_examples(ROOT / "data" / "raw" / language / f"{split}.jsonl")
        if split == "train":
            for seed in config["seeds"]:
                pool = parallel_pool(rows, pool_sizes["train"], seed)
                write_split(language, f"train_seed{seed}", balanced(pool, caps["train"], seed))
        else:
            pool = parallel_pool(rows, pool_sizes[split], EVAL_SEED)
            write_split(language, split, balanced(pool, caps[split], EVAL_SEED))


def main() -> None:
    config = yaml.safe_load((ROOT / "benchmark.yaml").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["all", *config["languages"]], default="all")
    args = parser.parse_args()
    languages = config["languages"] if args.language == "all" else [args.language]
    for language in languages:
        prepare(language, config)


if __name__ == "__main__":
    main()
