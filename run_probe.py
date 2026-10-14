#!/usr/bin/env python3
"""Run the fixed MASSIVE slot-word probing protocol for two languages."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
POOLINGS = (("first", 0), ("mean", 1))
GROUPS = ("one", "two", "three_plus")


def read_rows(language: str, name: str) -> list[dict]:
    path = ROOT / "data" / "prepared" / language / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class Encoder:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.cache: dict[tuple, tuple[np.ndarray, int]] = {}

    def encode(self, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        vectors, fertility = [], []
        with torch.no_grad():
            for row in rows:
                key = (row["locale"], row["utterance_id"], tuple(row["words"]), row["word_id"])
                if key not in self.cache:
                    encoded = self.tokenizer(
                        row["words"], is_split_into_words=True, return_tensors="pt",
                        truncation=True, max_length=512,
                    )
                    positions = [
                        i for i, word_id in enumerate(encoded.word_ids(0))
                        if word_id == row["word_id"]
                    ]
                    if not positions:
                        raise ValueError("target word was truncated")
                    states = self.model(**encoded, output_hidden_states=True).hidden_states
                    first = np.stack([layer[0, positions[0]].numpy() for layer in states])
                    mean = np.stack([layer[0, positions].mean(dim=0).numpy() for layer in states])
                    self.cache[key] = (np.stack([first, mean]), len(positions))
                vector, count = self.cache[key]
                vectors.append(vector)
                fertility.append(count)
        return np.stack(vectors), np.asarray(fertility)


def fit(x, y, seed: int) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, random_state=seed).fit(x, y)


def summary(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}


def group_of(count: int) -> str:
    return "one" if count == 1 else "two" if count == 2 else "three_plus"


def type_labels(words: list[str], labels: list[str]) -> tuple[dict[str, str], str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for word, label in zip(words, labels):
        counts[word][label] += 1
    majority = Counter(labels).most_common(1)[0][0]
    return {word: values.most_common(1)[0][0] for word, values in counts.items()}, majority


def type_predictions(words: list[str], mapping: dict[str, str], fallback: str) -> np.ndarray:
    return np.asarray([mapping.get(word, fallback) for word in words])


def control_mapping(words: list[str], labels: list[str], seed: int) -> dict[str, str]:
    values, counts = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(seed)
    return {word: str(rng.choice(values, p=counts / counts.sum())) for word in sorted(set(words))}


def run_language(language: str, encoder: Encoder, seeds: list[int]) -> dict:
    data = {}
    for name in [*seeds, "dev", "test"]:
        rows = read_rows(language, f"train_seed{name}" if isinstance(name, int) else name)
        x, fertility = encoder.encode(rows)
        data[name] = {
            "rows": rows,
            "x": x,
            "fertility": fertility,
            "labels": np.asarray([row["label"] for row in rows]),
            "words": [row["word"] for row in rows],
        }

    test = data["test"]
    groups = {}
    for group in GROUPS:
        labels = [
            label for label, count in zip(test["labels"], test["fertility"])
            if group_of(count) == group
        ]
        counts = Counter(labels)
        groups[group] = {
            "n": len(labels),
            "label_counts": dict(counts),
            "majority_accuracy": max(counts.values()) / len(labels) if labels else None,
        }

    result = {
        "audit": {
            "examples": {
                "train_per_seed": len(data[seeds[0]]["rows"]),
                "dev": len(data["dev"]["rows"]),
                "test": len(test["rows"]),
            },
            "test_label_counts": dict(Counter(test["labels"])),
            "mean_fertility_test": float(np.mean(test["fertility"])),
            "test_fertility_groups": groups,
        },
        "layer_profile": {},
        "selected_layer": {},
        "runs": [],
        "against_habit": {},
        "fertility": {},
        "test_predictions": {},
    }
    n_layers = test["x"].shape[2]

    for pooling, pool in POOLINGS:
        profile = []
        for layer in range(n_layers):
            task_scores, control_scores = [], []
            for seed in seeds:
                train, dev = data[seed], data["dev"]
                mapping = control_mapping(
                    train["words"] + dev["words"],
                    train["labels"].tolist() + dev["labels"].tolist(),
                    seed,
                )
                task_scores.append(
                    fit(train["x"][:, pool, layer], train["labels"], seed)
                    .score(dev["x"][:, pool, layer], dev["labels"])
                )
                control_scores.append(
                    fit(
                        train["x"][:, pool, layer],
                        np.asarray([mapping[word] for word in train["words"]]),
                        seed,
                    ).score(
                        dev["x"][:, pool, layer],
                        np.asarray([mapping[word] for word in dev["words"]]),
                    )
                )
            profile.append({
                "layer": layer,
                "dev_task": summary(task_scores),
                "dev_control": summary(control_scores),
                "dev_selectivity": summary([
                    task - control for task, control in zip(task_scores, control_scores)
                ]),
            })
        result["layer_profile"][pooling] = profile
        layer = int(np.argmax([row["dev_selectivity"]["mean"] for row in profile]))
        result["selected_layer"][pooling] = layer

        fertility = {group: {"task": []} for group in GROUPS}
        habit_task, habit_type, habit_n = [], [], []
        predictions = []
        for seed in seeds:
            train, dev = data[seed], data["dev"]
            train_x = np.concatenate([train["x"][:, pool, layer], dev["x"][:, pool, layer]])
            train_y = np.concatenate([train["labels"], dev["labels"]])
            train_words = train["words"] + dev["words"]
            task_probe = fit(train_x, train_y, seed)

            random_labels = control_mapping(train_words + test["words"], train_y.tolist(), seed)
            control_train = np.asarray([random_labels[word] for word in train_words])
            control_test = np.asarray([random_labels[word] for word in test["words"]])
            control_probe = fit(train_x, control_train, seed)

            type_map, fallback = type_labels(train_words, train_y.tolist())
            type_pred = type_predictions(test["words"], type_map, fallback)
            task_pred = task_probe.predict(test["x"][:, pool, layer])
            control_pred = control_probe.predict(test["x"][:, pool, layer])

            task_acc = float(np.mean(task_pred == test["labels"]))
            control_acc = float(np.mean(control_pred == control_test))
            result["runs"].append({
                "language": language,
                "pooling": pooling,
                "seed": seed,
                "selected_layer": layer,
                "task_accuracy": task_acc,
                "control_accuracy": control_acc,
                "selectivity": task_acc - control_acc,
                "majority_accuracy": float(max(Counter(test["labels"]).values()) / len(test["labels"])),
                "word_type_accuracy": float(np.mean(type_pred == test["labels"])),
            })

            known = np.asarray([word in type_map for word in test["words"]])
            against = known & (type_pred != test["labels"])
            habit_n.append(int(against.sum()))
            habit_task.append(
                float(np.mean(task_pred[against] == test["labels"][against]))
                if against.any() else float("nan")
            )
            habit_type.append(
                float(np.mean(type_pred[against] == test["labels"][against]))
                if against.any() else float("nan")
            )

            for group in GROUPS:
                mask = np.asarray([group_of(count) == group for count in test["fertility"]])
                if mask.any():
                    fertility[group]["task"].append(
                        float(np.mean(task_pred[mask] == test["labels"][mask]))
                    )

            if seed == seeds[0]:
                predictions = [{
                    "utterance_id": row["utterance_id"],
                    "sentence": " ".join(row["words"]),
                    "word": row["word"],
                    "slot_type": row["slot_type"],
                    "label": row["label"],
                    "predicted": str(predicted),
                    "subwords": int(count),
                    "against_habit": bool(is_against),
                } for row, predicted, count, is_against in zip(
                    test["rows"], task_pred, test["fertility"], against
                )]

        result["against_habit"][pooling] = {
            "n": habit_n,
            "task_accuracy": summary(habit_task),
            "word_type_accuracy": summary(habit_type),
        }
        result["fertility"][pooling] = {
            group: {
                "n": groups[group]["n"],
                "majority_accuracy": groups[group]["majority_accuracy"],
                "task_accuracy": summary(values["task"]),
            } if values["task"] else {"n": 0}
            for group, values in fertility.items()
        }
        result["test_predictions"][pooling] = predictions
    return result


def main() -> None:
    config = yaml.safe_load((ROOT / "benchmark.yaml").read_text())
    targets = [code for code in config["languages"] if code != "en"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=targets)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    encoder = Encoder(config["model"])
    seeds = list(config["seeds"])
    results = {
        "task": "MASSIVE slot-word detection",
        "language": args.language,
        "model": config["model"],
        "seeds": seeds,
        "languages": {},
    }
    for language in ("en", args.language):
        results["languages"][language] = run_language(language, encoder, seeds)
        print(f"{language}: selected layers {results['languages'][language]['selected_layer']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
