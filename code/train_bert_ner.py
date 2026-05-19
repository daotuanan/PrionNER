#!/usr/bin/env python3
"""Train and evaluate a BERT-family token classifier on CoNLL BIO data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


MODEL_ALIASES = {
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="final_data/fine/conll/train.conll")
    parser.add_argument("--test-file", default="final_data/fine/conll/test.conll")
    parser.add_argument("--model-name", required=True, help="HF model ID or alias.")
    parser.add_argument("--output-dir", required=True, help="Directory for model and evaluation outputs.")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-train-epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require all model/tokenizer files to exist in the local HF cache.",
    )
    return parser.parse_args()


def resolve_model_name(raw: str) -> str:
    return MODEL_ALIASES.get(raw.lower(), raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def collect_entity_labels(*tag_sets: Sequence[Sequence[str]]) -> List[str]:
    labels = set()
    for sequences in tag_sets:
        for tags in sequences:
            for tag in tags:
                if tag == "O":
                    continue
                labels.add(tag[2:])
    return sorted(labels)


def build_tag_vocab(entity_labels: Sequence[str]) -> List[str]:
    vocab = ["O"]
    for label in entity_labels:
        vocab.append(f"B-{label}")
    for label in entity_labels:
        vocab.append(f"I-{label}")
    return vocab


class TokenClassificationDataset(Dataset):
    def __init__(self, encodings, labels: Sequence[Sequence[int]]) -> None:
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def tokenize_and_align_labels(
    tokenizer,
    sentences_tokens: Sequence[Sequence[str]],
    sentences_tags: Sequence[Sequence[str]],
    label2id: Dict[str, int],
    max_length: int,
) -> TokenClassificationDataset:
    encodings = tokenizer(
        list(sentences_tokens),
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    aligned_labels: List[List[int]] = []

    for batch_index, tags in enumerate(sentences_tags):
        word_ids = encodings.word_ids(batch_index=batch_index)
        previous_word_id = None
        label_ids: List[int] = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(label2id[tags[word_id]])
            else:
                label_ids.append(-100)
            previous_word_id = word_id
        aligned_labels.append(label_ids)

    return TokenClassificationDataset(encodings, aligned_labels)


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


def decode_predictions(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    id2label: Dict[int, str],
) -> Tuple[List[List[str]], List[List[str]]]:
    pred_sequences: List[List[str]] = []
    gold_sequences: List[List[str]] = []
    pred_ids = predictions.argmax(axis=-1)

    for pred_row, gold_row in zip(pred_ids, label_ids):
        pred_tags: List[str] = []
        gold_tags: List[str] = []
        for pred_id, gold_id in zip(pred_row, gold_row):
            if gold_id == -100:
                continue
            pred_tags.append(id2label[int(pred_id)])
            gold_tags.append(id2label[int(gold_id)])
        pred_sequences.append(pred_tags)
        gold_sequences.append(gold_tags)

    return gold_sequences, pred_sequences


def trainer_metric_fn(id2label: Dict[int, str]):
    def compute(eval_pred) -> Dict[str, float]:
        predictions, label_ids = eval_pred
        gold_sequences, pred_sequences = decode_predictions(predictions, label_ids, id2label)
        metrics = compute_metrics_from_tags(gold_sequences, pred_sequences)
        return {
            "token_accuracy": float(metrics["token_accuracy"]),
            "entity_precision": float(metrics["entity_precision"]),
            "entity_recall": float(metrics["entity_recall"]),
            "entity_f1": float(metrics["entity_f1"]),
        }

    return compute


def write_predictions(
    output_path: Path,
    sentences_tokens: Sequence[Sequence[str]],
    gold_sequences: Sequence[Sequence[str]],
    pred_sequences: Sequence[Sequence[str]],
) -> None:
    lines: List[str] = []
    for tokens, gold_tags, pred_tags in zip(sentences_tokens, gold_sequences, pred_sequences):
        for token, gold_tag, pred_tag in zip(tokens, gold_tags, pred_tags):
            lines.append(f"{token}\t{gold_tag}\t{pred_tag}")
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_per_label_tsv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
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
    set_seed(args.seed)

    train_path = Path(args.train_file)
    test_path = Path(args.test_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tokens, train_tags = read_conll(train_path)
    test_tokens, test_tags = read_conll(test_path)
    entity_labels = collect_entity_labels(train_tags, test_tags)
    tag_list = build_tag_vocab(entity_labels)
    label2id = {label: index for index, label in enumerate(tag_list)}
    id2label = {index: label for label, index in label2id.items()}

    model_name = resolve_model_name(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.local_files_only)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(tag_list),
        id2label=id2label,
        label2id=label2id,
        local_files_only=args.local_files_only,
    )

    train_dataset = tokenize_and_align_labels(tokenizer, train_tokens, train_tags, label2id, args.max_length)
    test_dataset = tokenize_and_align_labels(tokenizer, test_tokens, test_tags, label2id, args.max_length)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=trainer_metric_fn(id2label),
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))

    predictions = trainer.predict(test_dataset)
    gold_sequences, pred_sequences = decode_predictions(predictions.predictions, predictions.label_ids, id2label)
    test_metrics = compute_metrics_from_tags(gold_sequences, pred_sequences)

    run_config = {
        "model_name": model_name,
        "train_file": str(train_path),
        "test_file": str(test_path),
        "output_dir": str(output_dir),
        "max_length": args.max_length,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "seed": args.seed,
        "local_files_only": args.local_files_only,
        "tag_list": tag_list,
    }

    metrics_payload = {
        "config": run_config,
        "train_runtime": train_result.metrics,
        "eval_metrics": {
            key: value for key, value in test_metrics.items() if key != "per_label"
        },
        "trainer_predict_metrics": predictions.metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_per_label_tsv(output_dir / "per_label_metrics.tsv", test_metrics["per_label"])
    write_predictions(output_dir / "test_predictions.conll", test_tokens, gold_sequences, pred_sequences)
    (output_dir / "train_log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "model_name": model_name,
        "entity_precision": test_metrics["entity_precision"],
        "entity_recall": test_metrics["entity_recall"],
        "entity_f1": test_metrics["entity_f1"],
        "token_accuracy": test_metrics["token_accuracy"],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
