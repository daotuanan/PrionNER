#!/usr/bin/env python3
"""Evaluate CoNLL BIO predictions with the same metrics used by the BERT baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-file", required=True)
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--source-name", default="")
    return parser.parse_args()


def read_conll(path: Path) -> Tuple[List[List[str]], List[List[str]]]:
    sentences_tokens: List[List[str]] = []
    sentences_tags: List[List[str]] = []
    cur_tokens: List[str] = []
    cur_tags: List[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            if cur_tokens:
                sentences_tokens.append(cur_tokens)
                sentences_tags.append(cur_tags)
                cur_tokens = []
                cur_tags = []
            continue
        token, tag = raw_line.rsplit("\t", 1)
        cur_tokens.append(token)
        cur_tags.append(tag)

    if cur_tokens:
        sentences_tokens.append(cur_tokens)
        sentences_tags.append(cur_tags)

    if not sentences_tokens:
        raise SystemExit(f"No sequences found in {path}")
    return sentences_tokens, sentences_tags


def close_entity(entities: List[Tuple[str, int, int]], active_label: str | None, start: int | None, end: int) -> None:
    if active_label is not None and start is not None:
        entities.append((active_label, start, end))


def bio_to_entities(tags: Sequence[str]) -> List[Tuple[str, int, int]]:
    entities: List[Tuple[str, int, int]] = []
    active_label: str | None = None
    start: int | None = None

    for index, tag in enumerate(tags):
        if tag == "O":
            close_entity(entities, active_label, start, index)
            active_label = None
            start = None
            continue

        prefix, label = tag.split("-", 1)
        if prefix == "B":
            close_entity(entities, active_label, start, index)
            active_label = label
            start = index
            continue

        if prefix == "I" and active_label == label and start is not None:
            continue

        close_entity(entities, active_label, start, index)
        active_label = label
        start = index

    close_entity(entities, active_label, start, len(tags))
    return entities


def compute_metrics_from_tags(
    gold_sequences: Sequence[Sequence[str]],
    pred_sequences: Sequence[Sequence[str]],
) -> Dict[str, object]:
    total_tokens = 0
    correct_tokens = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    per_label = Counter()
    per_label_tp = Counter()
    per_label_fp = Counter()
    per_label_fn = Counter()

    for gold_tags, pred_tags in zip(gold_sequences, pred_sequences):
        for gold_tag, pred_tag in zip(gold_tags, pred_tags):
            total_tokens += 1
            correct_tokens += int(gold_tag == pred_tag)

        gold_entities = set(bio_to_entities(gold_tags))
        pred_entities = set(bio_to_entities(pred_tags))
        overlap = gold_entities & pred_entities

        true_positive += len(overlap)
        false_positive += len(pred_entities - gold_entities)
        false_negative += len(gold_entities - pred_entities)

        for label, _, _ in gold_entities:
            per_label[label] += 1
        for label, _, _ in overlap:
            per_label_tp[label] += 1
        for label, _, _ in pred_entities - gold_entities:
            per_label_fp[label] += 1
        for label, _, _ in gold_entities - pred_entities:
            per_label_fn[label] += 1

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    token_accuracy = correct_tokens / total_tokens if total_tokens else 0.0

    labels = sorted(set(per_label) | set(per_label_tp) | set(per_label_fp) | set(per_label_fn))
    per_label_rows: List[Dict[str, object]] = []
    for label in labels:
        tp = per_label_tp[label]
        fp = per_label_fp[label]
        fn = per_label_fn[label]
        label_precision = tp / (tp + fp) if (tp + fp) else 0.0
        label_recall = tp / (tp + fn) if (tp + fn) else 0.0
        label_f1 = (2 * label_precision * label_recall / (label_precision + label_recall)) if (label_precision + label_recall) else 0.0
        per_label_rows.append(
            {
                "label": label,
                "gold_count": per_label[label],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": label_precision,
                "recall": label_recall,
                "f1": label_f1,
            }
        )

    return {
        "token_accuracy": token_accuracy,
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "entity_tp": true_positive,
        "entity_fp": false_positive,
        "entity_fn": false_negative,
        "per_label": per_label_rows,
    }


def write_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    header = ["label", "gold_count", "tp", "fp", "fn", "precision", "recall", "f1"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["label"]),
                    str(row["gold_count"]),
                    str(row["tp"]),
                    str(row["fp"]),
                    str(row["fn"]),
                    f"{float(row['precision']):.6f}",
                    f"{float(row['recall']):.6f}",
                    f"{float(row['f1']):.6f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_tokens, gold_tags = read_conll(Path(args.gold_file))
    pred_tokens, pred_tags = read_conll(Path(args.pred_file))

    if len(gold_tokens) != len(pred_tokens):
        raise SystemExit(f"Sentence count mismatch: gold={len(gold_tokens)} pred={len(pred_tokens)}")
    for index, (gold_sentence, pred_sentence) in enumerate(zip(gold_tokens, pred_tokens), start=1):
        if gold_sentence != pred_sentence:
            raise SystemExit(f"Token mismatch at sentence {index}")

    metrics = compute_metrics_from_tags(gold_tags, pred_tags)
    metrics_json = {
        "config": {
            "model_name": args.model_name,
            "train_file": "",
            "test_file": str(Path(args.gold_file).resolve()),
            "prediction_file": str(Path(args.pred_file).resolve()),
            "output_dir": str(output_dir.resolve()),
            "source_name": args.source_name,
        },
        "eval_metrics": {
            key: metrics[key]
            for key in (
                "token_accuracy",
                "entity_precision",
                "entity_recall",
                "entity_f1",
                "entity_tp",
                "entity_fp",
                "entity_fn",
            )
        },
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_tsv(output_dir / "per_label_metrics.tsv", metrics["per_label"])
    print(json.dumps(metrics_json["eval_metrics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
