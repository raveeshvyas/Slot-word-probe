#!/usr/bin/env python3
"""Check that a result file has the fields the report and grading rely on."""
from __future__ import annotations

import json
import sys
from pathlib import Path


LANGUAGE_FIELDS = {"audit", "layer_profile", "selected_layer", "runs", "against_habit",
                   "fertility", "test_predictions"}
RUN_FIELDS = {"language", "pooling", "seed", "selected_layer", "task_accuracy", "control_accuracy",
              "selectivity", "majority_accuracy", "word_type_accuracy"}
POOLINGS = ("first", "mean")
GROUPS = ("one", "two", "three_plus")


def fail(message: str) -> None:
    raise SystemExit(f"invalid result file: {message}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_results.py results/LANGUAGE.json")
    result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    for field in ("language", "model", "seeds", "languages"):
        if field not in result:
            fail(f"missing top-level field '{field}'")
    if len(result["seeds"]) != 5:
        fail("expected the five benchmark seeds")
    for language in ("en", result["language"]):
        block = result["languages"].get(language)
        if block is None:
            fail(f"missing language '{language}'")
        missing = LANGUAGE_FIELDS - block.keys()
        if missing:
            fail(f"{language}: missing {sorted(missing)}")
        for pooling in POOLINGS:
            if pooling not in block["layer_profile"] or pooling not in block["selected_layer"]:
                fail(f"{language}: missing layer profile or selected layer for '{pooling}'")
            if pooling not in block["against_habit"]:
                fail(f"{language}: missing against-habit results for '{pooling}'")
            if not all(group in block["fertility"].get(pooling, {}) for group in GROUPS):
                fail(f"{language}: missing fertility groups for '{pooling}'")
            runs = [r for r in block["runs"] if r["pooling"] == pooling]
            if sorted(r["seed"] for r in runs) != sorted(result["seeds"]):
                fail(f"{language}/{pooling}: expected one run per seed")
            for run in runs:
                if RUN_FIELDS - run.keys():
                    fail(f"{language}/{pooling}: run missing {sorted(RUN_FIELDS - run.keys())}")
    print("result file has the expected benchmark fields")


if __name__ == "__main__":
    main()
