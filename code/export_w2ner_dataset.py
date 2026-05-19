#!/usr/bin/env python3
"""Export BRAT NER annotations into the JSON format expected by W2NER."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from brat_to_json_conll import Document, Entity, Span, iter_line_tokens, load_documents, spans_overlap


W2NER_DEFAULTS = {
    "dist_emb_size": 20,
    "type_emb_size": 20,
    "lstm_hid_size": 512,
    "conv_hid_size": 128,
    "bert_hid_size": 768,
    "biaffine_size": 512,
    "ffnn_hid_size": 384,
    "dilation": [1, 2, 3, 4],
    "emb_dropout": 0.5,
    "conv_dropout": 0.5,
    "out_dropout": 0.33,
    "epochs": 10,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "clip_grad_norm": 5.0,
    "bert_name": "dmis-lab/biobert-v1.1",
    "bert_learning_rate": 5e-6,
    "warm_factor": 0.1,
    "use_bert_last_4_layers": False,
    "seed": 123,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-brat-dir", default="final_data/fine/brat/train")
    parser.add_argument("--test-brat-dir", default="final_data/fine/brat/test")
    parser.add_argument(
        "--dev-brat-dir",
        help="Optional BRAT directory for a separate dev set. If omitted, dev is split from train documents.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where train.json/dev.json/test.json and metadata will be written.",
    )
    parser.add_argument(
        "--dataset-name",
        default="prion_fine_w2ner",
        help="Dataset identifier to place into the generated W2NER config.",
    )
    parser.add_argument(
        "--dev-doc-fraction",
        type=float,
        default=0.1,
        help="Fraction of training documents to hold out for dev when --dev-brat-dir is omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/dev document splitting.",
    )
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument(
        "--config-file",
        help="Optional path for an upstream-compatible W2NER config JSON.",
    )
    parser.add_argument(
        "--save-path",
        help="Optional model checkpoint path to place into the generated config.",
    )
    parser.add_argument(
        "--predict-path",
        help="Optional prediction JSON path to place into the generated config.",
    )
    parser.add_argument("--bert-name", default=W2NER_DEFAULTS["bert_name"])
    parser.add_argument("--dist-emb-size", type=int, default=W2NER_DEFAULTS["dist_emb_size"])
    parser.add_argument("--type-emb-size", type=int, default=W2NER_DEFAULTS["type_emb_size"])
    parser.add_argument("--lstm-hid-size", type=int, default=W2NER_DEFAULTS["lstm_hid_size"])
    parser.add_argument("--conv-hid-size", type=int, default=W2NER_DEFAULTS["conv_hid_size"])
    parser.add_argument("--bert-hid-size", type=int, default=W2NER_DEFAULTS["bert_hid_size"])
    parser.add_argument("--biaffine-size", type=int, default=W2NER_DEFAULTS["biaffine_size"])
    parser.add_argument("--ffnn-hid-size", type=int, default=W2NER_DEFAULTS["ffnn_hid_size"])
    parser.add_argument(
        "--dilation",
        type=int,
        nargs="+",
        default=list(W2NER_DEFAULTS["dilation"]),
    )
    parser.add_argument("--emb-dropout", type=float, default=W2NER_DEFAULTS["emb_dropout"])
    parser.add_argument("--conv-dropout", type=float, default=W2NER_DEFAULTS["conv_dropout"])
    parser.add_argument("--out-dropout", type=float, default=W2NER_DEFAULTS["out_dropout"])
    parser.add_argument("--epochs", type=int, default=W2NER_DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=W2NER_DEFAULTS["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=W2NER_DEFAULTS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=W2NER_DEFAULTS["weight_decay"])
    parser.add_argument("--clip-grad-norm", type=float, default=W2NER_DEFAULTS["clip_grad_norm"])
    parser.add_argument(
        "--bert-learning-rate",
        type=float,
        default=W2NER_DEFAULTS["bert_learning_rate"],
    )
    parser.add_argument("--warm-factor", type=float, default=W2NER_DEFAULTS["warm_factor"])
    parser.add_argument(
        "--use-bert-last-4-layers",
        action="store_true",
        default=W2NER_DEFAULTS["use_bert_last_4_layers"],
    )
    parser.add_argument(
        "--config-seed",
        type=int,
        default=W2NER_DEFAULTS["seed"],
        help="Seed to place into the generated W2NER config.",
    )
    return parser.parse_args()


def entity_within_line(entity: Entity, line_span: Span) -> bool:
    return all(line_span.start <= span.start and span.end <= line_span.end for span in entity.spans)


def entities_overlap(left: Entity, right: Entity) -> bool:
    return any(spans_overlap(left_span, right_span) for left_span in left.spans for right_span in right.spans)


def dedupe_preserve_order(values: Iterable[int]) -> List[int]:
    seen = set()
    result: List[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def token_indexes_for_span(token_spans: Sequence[Span], span: Span) -> List[int]:
    return [index for index, token_span in enumerate(token_spans) if spans_overlap(token_span, span)]


def serialize_entity(
    entity: Entity,
    token_spans: Sequence[Span],
    line_span: Span,
    non_flat_entity_ids: frozenset[str],
) -> Dict[str, object]:
    token_indexes: List[int] = []
    for span in entity.spans:
        current = token_indexes_for_span(token_spans, span)
        if not current:
            raise ValueError(
                f"Entity {entity.entity_id} ({entity.label}) has no token overlap for span "
                f"{span.start}-{span.end}."
            )
        token_indexes.extend(current)

    relative_spans = [
        {"start": span.start - line_span.start, "end": span.end - line_span.start}
        for span in entity.spans
    ]
    return {
        "id": entity.entity_id,
        "type": entity.label,
        "index": dedupe_preserve_order(token_indexes),
        "text": entity.text,
        "spans": relative_spans,
        "is_discontinuous": len(entity.spans) > 1,
        "is_non_flat": entity.entity_id in non_flat_entity_ids,
    }


def build_instances(documents: Sequence[Document]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    instances: List[Dict[str, object]] = []
    label_counts: Dict[str, int] = {}
    total_entities = 0
    discontinuous_entities = 0
    non_flat_entities = 0

    for document in documents:
        line_records = list(iter_line_tokens(document.text))
        for line_index, (tokens, token_spans, line_span) in enumerate(line_records):
            line_entities = [entity for entity in document.entities if entity_within_line(entity, line_span)]
            overlapping_ids: set[str] = set()
            for left_index, left in enumerate(line_entities):
                for right in line_entities[left_index + 1:]:
                    if entities_overlap(left, right):
                        overlapping_ids.add(left.entity_id)
                        overlapping_ids.add(right.entity_id)
            non_flat_entity_ids = frozenset(
                overlapping_ids | {entity.entity_id for entity in line_entities if len(entity.spans) > 1}
            )

            ner = [
                serialize_entity(entity, token_spans, line_span, non_flat_entity_ids)
                for entity in line_entities
            ]
            ner.sort(key=lambda item: (item["index"][0] if item["index"] else -1, len(item["index"]), item["type"]))

            for entity in ner:
                label = str(entity["type"])
                label_counts[label] = label_counts.get(label, 0) + 1
                total_entities += 1
                discontinuous_entities += int(bool(entity["is_discontinuous"]))
                non_flat_entities += int(bool(entity["is_non_flat"]))

            instances.append(
                {
                    "id": f"{document.doc_id}:{line_index + 1}",
                    "doc_id": document.doc_id,
                    "line_index": line_index,
                    "sentence": tokens,
                    "text": document.text[line_span.start:line_span.end],
                    "token_offsets": [
                        {"start": span.start - line_span.start, "end": span.end - line_span.start}
                        for span in token_spans
                    ],
                    "ner": ner,
                }
            )

    stats = {
        "documents": len(documents),
        "instances": len(instances),
        "entities": total_entities,
        "discontinuous_entities": discontinuous_entities,
        "non_flat_entities": non_flat_entities,
        "label_counts": dict(sorted(label_counts.items())),
    }
    return instances, stats


def split_train_dev(
    train_documents: Sequence[Document],
    dev_documents: Sequence[Document] | None,
    dev_doc_fraction: float,
    seed: int,
) -> Tuple[List[Document], List[Document]]:
    if dev_documents is not None:
        return list(train_documents), list(dev_documents)

    if not 0.0 <= dev_doc_fraction < 1.0:
        raise SystemExit(f"--dev-doc-fraction must be in [0, 1), got {dev_doc_fraction}")

    shuffled = list(train_documents)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1 or dev_doc_fraction == 0.0:
        return sorted(shuffled, key=lambda document: document.doc_id), []

    dev_count = max(1, round(len(shuffled) * dev_doc_fraction))
    dev_count = min(dev_count, len(shuffled) - 1)
    dev_split = sorted(shuffled[:dev_count], key=lambda document: document.doc_id)
    train_split = sorted(shuffled[dev_count:], key=lambda document: document.doc_id)
    return train_split, dev_split


def build_config(args: argparse.Namespace, config_file: Path, output_dir: Path) -> Dict[str, object]:
    save_path = Path(args.save_path) if args.save_path else output_dir / "model.pt"
    predict_path = Path(args.predict_path) if args.predict_path else output_dir / "output.json"
    return {
        "dataset": args.dataset_name,
        "save_path": str(save_path.resolve()),
        "predict_path": str(predict_path.resolve()),
        "dist_emb_size": args.dist_emb_size,
        "type_emb_size": args.type_emb_size,
        "lstm_hid_size": args.lstm_hid_size,
        "conv_hid_size": args.conv_hid_size,
        "bert_hid_size": args.bert_hid_size,
        "biaffine_size": args.biaffine_size,
        "ffnn_hid_size": args.ffnn_hid_size,
        "dilation": list(args.dilation),
        "emb_dropout": args.emb_dropout,
        "conv_dropout": args.conv_dropout,
        "out_dropout": args.out_dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "clip_grad_norm": args.clip_grad_norm,
        "bert_name": args.bert_name,
        "bert_learning_rate": args.bert_learning_rate,
        "warm_factor": args.warm_factor,
        "use_bert_last_4_layers": args.use_bert_last_4_layers,
        "seed": args.config_seed,
        "_config_file": str(config_file.resolve()),
        "_dataset_dir": str(output_dir.resolve()),
    }


def write_json(path: Path, payload: object, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding=encoding)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_documents = load_documents(Path(args.train_brat_dir), args.encoding)
    test_documents = load_documents(Path(args.test_brat_dir), args.encoding)
    dev_documents = (
        load_documents(Path(args.dev_brat_dir), args.encoding)
        if args.dev_brat_dir
        else None
    )

    train_split, dev_split = split_train_dev(
        train_documents=train_documents,
        dev_documents=dev_documents,
        dev_doc_fraction=args.dev_doc_fraction,
        seed=args.seed,
    )

    train_instances, train_stats = build_instances(train_split)
    dev_instances, dev_stats = build_instances(dev_split)
    test_instances, test_stats = build_instances(test_documents)

    write_json(output_dir / "train.json", train_instances, args.encoding)
    write_json(output_dir / "dev.json", dev_instances, args.encoding)
    write_json(output_dir / "test.json", test_instances, args.encoding)

    manifest = {
        "dataset_name": args.dataset_name,
        "train_doc_ids": [document.doc_id for document in train_split],
        "dev_doc_ids": [document.doc_id for document in dev_split],
        "test_doc_ids": [document.doc_id for document in test_documents],
        "train_stats": train_stats,
        "dev_stats": dev_stats,
        "test_stats": test_stats,
    }
    write_json(output_dir / "manifest.json", manifest, args.encoding)

    if args.config_file:
        config_file = Path(args.config_file)
        config = build_config(args, config_file=config_file, output_dir=output_dir)
        write_json(config_file, config, args.encoding)
        print(f"Wrote config: {config_file}")

    print(f"Wrote W2NER dataset: {output_dir}")
    print(
        "Split stats: "
        f"train={train_stats['documents']} docs/{train_stats['instances']} instances, "
        f"dev={dev_stats['documents']} docs/{dev_stats['instances']} instances, "
        f"test={test_stats['documents']} docs/{test_stats['instances']} instances"
    )
    print(
        "Entity stats: "
        f"train={train_stats['entities']} total/{train_stats['non_flat_entities']} non-flat, "
        f"dev={dev_stats['entities']} total/{dev_stats['non_flat_entities']} non-flat, "
        f"test={test_stats['entities']} total/{test_stats['non_flat_entities']} non-flat"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
