#!/usr/bin/env python3
"""Convert W2NER prediction output into BRAT .ann files for repo-wide evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from brat_to_json_conll import iter_line_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w2ner-output", required=True)
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--gold-brat-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoding", default="utf-8")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def merge_token_indexes(indexes: Sequence[int], token_offsets: Sequence[Dict[str, int]]) -> List[Tuple[int, int]]:
    if not indexes:
        return []

    sorted_indexes = sorted(indexes)
    spans: List[Tuple[int, int]] = []
    start = token_offsets[sorted_indexes[0]]["start"]
    end = token_offsets[sorted_indexes[0]]["end"]

    for previous, current in zip(sorted_indexes, sorted_indexes[1:]):
        current_start = token_offsets[current]["start"]
        current_end = token_offsets[current]["end"]
        if current == previous + 1:
            end = current_end
            continue
        spans.append((start, end))
        start = current_start
        end = current_end
    spans.append((start, end))
    return spans


def span_texts(text: str, spans: Iterable[Tuple[int, int]]) -> str:
    return " ".join(text[start:end] for start, end in spans)


def main() -> int:
    args = parse_args()
    predictions = load_json(Path(args.w2ner_output))
    dataset_items = load_json(Path(args.dataset_json))
    if len(predictions) != len(dataset_items):
        raise SystemExit(
            f"Prediction/item length mismatch: {len(predictions)} predictions vs {len(dataset_items)} dataset items"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_dir = Path(args.gold_brat_dir)

    by_doc: Dict[str, List[str]] = {}
    next_tid: Dict[str, int] = {}

    for prediction, dataset_item in zip(predictions, dataset_items):
        doc_id = str(dataset_item["doc_id"])
        line_index = int(dataset_item["line_index"])
        if prediction.get("id") != dataset_item.get("id"):
            raise SystemExit(f"ID mismatch: prediction={prediction.get('id')} dataset={dataset_item.get('id')}")

        text_path = gold_dir / f"{doc_id}.txt"
        text = text_path.read_text(encoding=args.encoding)
        line_records = list(iter_line_tokens(text))
        _, _, line_span = line_records[line_index]

        lines = by_doc.setdefault(doc_id, [])
        next_tid.setdefault(doc_id, 1)

        sentence_text = dataset_item["text"]
        token_offsets = dataset_item["token_offsets"]
        for entity in prediction.get("entity", []):
            indexes = entity.get("index")
            label = entity.get("type")
            if not isinstance(indexes, list) or not isinstance(label, str):
                continue
            rel_spans = merge_token_indexes(indexes, token_offsets)
            abs_spans = [
                (line_span.start + span_start, line_span.start + span_end)
                for span_start, span_end in rel_spans
            ]
            mention_text = span_texts(sentence_text, rel_spans)
            span_field = ";".join(f"{start} {end}" for start, end in abs_spans)
            tid = f"T{next_tid[doc_id]}"
            next_tid[doc_id] += 1
            lines.append(f"{tid}\t{label} {span_field}\t{mention_text}")

    for txt_path in sorted(gold_dir.glob("*.txt")):
        doc_id = txt_path.stem
        (output_dir / txt_path.name).write_text(txt_path.read_text(encoding=args.encoding), encoding=args.encoding)
        ann_lines = by_doc.get(doc_id, [])
        ann_text = "\n".join(ann_lines) + ("\n" if ann_lines else "")
        (output_dir / f"{doc_id}.ann").write_text(ann_text, encoding=args.encoding)

    print(f"Wrote BRAT predictions: {output_dir}")
    print(f"Documents: {len(list(gold_dir.glob('*.txt')))}")
    print(f"Predicted entities: {sum(len(lines) for lines in by_doc.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
