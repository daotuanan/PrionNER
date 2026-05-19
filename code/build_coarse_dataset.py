#!/usr/bin/env python3
"""Build coarse-grained BRAT/JSON/CoNLL datasets from the fine-grained BRAT gold set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from brat_to_json_conll import parse_ann


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-brat-dir", default="final_data/fine/brat/test")
    parser.add_argument("--coarse-brat-dir", default="final_data/coarse/brat/test")
    parser.add_argument("--schema-file", default="prion_ner_entity_definitions_fine.json")
    parser.add_argument("--encoding", default="utf-8")
    return parser.parse_args()


def load_label_map(schema_path: Path) -> Dict[str, str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    label_map: Dict[str, str] = {}
    for entity_type in schema.get("entity_types", []):
        coarse = str(entity_type.get("coarse_type", "")).strip()
        if not coarse:
            continue
        label_map[coarse] = coarse
        for fine_type in entity_type.get("fine_types", []):
            fine_label = str(fine_type.get("fine_type", "")).strip()
            brat_label = str(fine_type.get("brat_label", fine_label)).strip()
            if fine_label:
                label_map[fine_label] = coarse
            if brat_label:
                label_map[brat_label] = coarse
    return label_map


def span_text(spans: Tuple[object, ...]) -> str:
    return ";".join(f"{span.start} {span.end}" for span in spans)


def main() -> int:
    args = parse_args()
    fine_dir = Path(args.fine_brat_dir)
    coarse_dir = Path(args.coarse_brat_dir)
    coarse_dir.mkdir(parents=True, exist_ok=True)
    label_map = load_label_map(Path(args.schema_file))

    converted_docs = 0
    converted_entities = 0
    for txt_path in sorted(fine_dir.glob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            raise SystemExit(f"Missing fine annotation file for {txt_path}")
        text = txt_path.read_text(encoding=args.encoding)
        entities = parse_ann(ann_path, text, args.encoding)

        coarse_ann_lines = []
        for entity in entities:
            coarse_label = label_map.get(entity.label)
            if coarse_label is None:
                raise SystemExit(f"No coarse label mapping found for {entity.label!r} in {ann_path}")
            coarse_ann_lines.append(
                f"{entity.entity_id}\t{coarse_label} {span_text(entity.spans)}\t{entity.text}"
            )
        (coarse_dir / txt_path.name).write_text(text, encoding=args.encoding)
        (coarse_dir / ann_path.name).write_text(
            "\n".join(coarse_ann_lines) + ("\n" if coarse_ann_lines else ""),
            encoding=args.encoding,
        )
        converted_docs += 1
        converted_entities += len(entities)

    print(f"Built coarse BRAT dataset: {converted_docs} docs, {converted_entities} entities -> {coarse_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
