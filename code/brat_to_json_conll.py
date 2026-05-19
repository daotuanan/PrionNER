#!/usr/bin/env python3
"""Convert BRAT text-bound annotations into JSON and CoNLL BIO formats."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TOKEN_RE = re.compile(r"\S+")
TEXTBOUND_RE = re.compile(r"^(T\d+)\t(\S+) ([0-9]+ [0-9]+(?:;[0-9]+ [0-9]+)*)\t(.*)$")


@dataclass(frozen=True)
class Span:
    start: int
    end: int


@dataclass(frozen=True)
class Entity:
    entity_id: str
    label: str
    spans: Tuple[Span, ...]
    text: str

    @property
    def start(self) -> int:
        return min(span.start for span in self.spans)

    @property
    def end(self) -> int:
        return max(span.end for span in self.spans)

    @property
    def is_discontinuous(self) -> bool:
        return len(self.spans) > 1


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    entities: Tuple[Entity, ...]
    txt_path: Path
    ann_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="final_data/fine/brat/test",
        help="Directory containing paired .txt and .ann BRAT files.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("json", "conll", "both"),
        default=["both"],
        help="Output format(s) to generate. Default: both.",
    )
    parser.add_argument(
        "--json-dir",
        help="Directory for per-document JSON files. Defaults to final_data/fine/json/<split_name>.",
    )
    parser.add_argument(
        "--conll-file",
        help="Path for CoNLL BIO output. Defaults to final_data/fine/conll/<split_name>.conll.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for input and output files.",
    )
    return parser.parse_args()


def normalize_formats(raw_formats: Sequence[str]) -> Tuple[str, ...]:
    normalized = set(raw_formats)
    if "both" in normalized:
        normalized.discard("both")
        normalized.update({"json", "conll"})
    return tuple(sorted(normalized))


def parse_span_list(raw: str) -> Tuple[Span, ...]:
    spans: List[Span] = []
    for chunk in raw.split(";"):
        start_str, end_str = chunk.split()
        start = int(start_str)
        end = int(end_str)
        if start >= end:
            raise ValueError(f"Invalid span {chunk!r}.")
        spans.append(Span(start=start, end=end))
    return tuple(spans)


def extract_spanned_text(text: str, spans: Sequence[Span]) -> str:
    return " ".join(text[span.start:span.end] for span in spans)


def parse_ann(path: Path, text: str, encoding: str) -> Tuple[Entity, ...]:
    entities: List[Entity] = []
    for line_number, line in enumerate(path.read_text(encoding=encoding).splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        match = TEXTBOUND_RE.match(stripped)
        if not match:
            raise ValueError(f"{path}:{line_number}: Unsupported annotation line: {line}")
        entity_id, label, raw_spans, annotated_text = match.groups()
        spans = parse_span_list(raw_spans)
        extracted = extract_spanned_text(text, spans)
        if extracted != annotated_text:
            raise ValueError(
                f"{path}:{line_number}: Text mismatch for {entity_id}: "
                f"annotated={annotated_text!r}, extracted={extracted!r}"
            )
        entities.append(Entity(entity_id=entity_id, label=label, spans=spans, text=annotated_text))
    entities.sort(key=lambda entity: (entity.start, entity.end, entity.label, entity.entity_id))
    return tuple(entities)


def load_documents(input_dir: Path, encoding: str) -> Tuple[Document, ...]:
    txt_paths = sorted(input_dir.glob("*.txt"))
    if not txt_paths:
        raise SystemExit(f"No .txt files found in {input_dir}")

    documents: List[Document] = []
    for txt_path in txt_paths:
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            raise SystemExit(f"Missing paired annotation file for {txt_path}: {ann_path}")
        text = txt_path.read_text(encoding=encoding)
        entities = parse_ann(ann_path, text, encoding)
        documents.append(
            Document(
                doc_id=txt_path.stem,
                text=text,
                entities=entities,
                txt_path=txt_path,
                ann_path=ann_path,
            )
        )
    return tuple(documents)


def json_ready_entity(entity: Entity) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "id": entity.entity_id,
        "label": entity.label,
        "start": entity.start,
        "end": entity.end,
        "text": entity.text,
        "spans": [{"start": span.start, "end": span.end} for span in entity.spans],
    }
    if entity.is_discontinuous:
        payload["is_discontinuous"] = True
    return payload


def write_json_documents(documents: Sequence[Document], output_dir: Path, encoding: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for document in documents:
        payload = {
            "doc_id": document.doc_id,
            "text": document.text,
            "entities": [json_ready_entity(entity) for entity in document.entities],
        }
        out_path = output_dir / f"{document.doc_id}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding=encoding)


def iter_line_tokens(text: str) -> Iterable[Tuple[List[str], List[Span], Span]]:
    offset = 0
    for line in text.splitlines(keepends=True):
        line_body = line[:-1] if line.endswith("\n") else line
        line_span = Span(offset, offset + len(line_body))
        tokens: List[str] = []
        spans: List[Span] = []
        for match in TOKEN_RE.finditer(line_body):
            tokens.append(match.group(0))
            spans.append(Span(offset + match.start(), offset + match.end()))
        if tokens:
            yield tokens, spans, line_span
        offset += len(line)


def spans_overlap(left: Span, right: Span) -> bool:
    return left.start < right.end and right.start < left.end


def chunk_sort_key(item: Tuple[int, str, str, Span]) -> Tuple[int, int, int, str]:
    _, entity_id, label, span = item
    return (-(span.end - span.start), span.start, span.end, f"{label}:{entity_id}")


def entity_overlaps_line(entity: Entity, line_span: Span) -> bool:
    return any(spans_overlap(span, line_span) for span in entity.spans)


def assign_bio_tags(
    token_spans: Sequence[Span],
    entities: Sequence[Entity],
) -> Tuple[List[str], List[str]]:
    tags = ["O"] * len(token_spans)
    owners: List[str | None] = [None] * len(token_spans)
    conflicts: List[str] = []

    chunks: List[Tuple[int, str, str, Span]] = []
    for entity_index, entity in enumerate(entities):
        for span in entity.spans:
            chunks.append((entity_index, entity.entity_id, entity.label, span))
    chunks.sort(key=chunk_sort_key)

    for _, entity_id, label, span in chunks:
        token_indexes = [
            index for index, token_span in enumerate(token_spans)
            if spans_overlap(token_span, span)
        ]
        if not token_indexes:
            conflicts.append(f"{entity_id}:{label}@{span.start}-{span.end}:no-token-overlap")
            continue
        if any(owners[index] is not None for index in token_indexes):
            conflicts.append(f"{entity_id}:{label}@{span.start}-{span.end}:token-conflict")
            continue
        for offset, token_index in enumerate(token_indexes):
            tags[token_index] = ("B-" if offset == 0 else "I-") + label
            owners[token_index] = entity_id

    return tags, conflicts


def write_conll(documents: Sequence[Document], output_path: Path, encoding: str) -> Tuple[int, List[str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    conflict_count = 0
    conflict_examples: List[str] = []

    for document in documents:
        for tokens, token_spans, line_span in iter_line_tokens(document.text):
            line_entities = [
                entity for entity in document.entities
                if entity_overlaps_line(entity, line_span)
            ]
            tags, conflicts = assign_bio_tags(token_spans, line_entities)
            conflict_count += len(conflicts)
            for conflict in conflicts[:5]:
                if len(conflict_examples) < 20:
                    conflict_examples.append(f"{document.doc_id}:{conflict}")
            for token, tag in zip(tokens, tags):
                lines.append(f"{token}\t{tag}")
            lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding=encoding)
    return conflict_count, conflict_examples


def default_json_dir(input_dir: Path) -> Path:
    split_name = input_dir.name
    return Path("final_data/fine/json") / split_name


def default_conll_file(input_dir: Path) -> Path:
    split_name = input_dir.name
    return Path("final_data/fine/conll") / f"{split_name}.conll"


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    formats = normalize_formats(args.formats)
    json_dir = Path(args.json_dir) if args.json_dir else default_json_dir(input_dir)
    conll_file = Path(args.conll_file) if args.conll_file else default_conll_file(input_dir)

    documents = load_documents(input_dir, args.encoding)

    if "json" in formats:
        write_json_documents(documents, json_dir, args.encoding)

    conflict_count = 0
    conflict_examples: List[str] = []
    if "conll" in formats:
        conflict_count, conflict_examples = write_conll(documents, conll_file, args.encoding)

    entity_count = sum(len(document.entities) for document in documents)
    discontinuous_count = sum(
        1 for document in documents for entity in document.entities if entity.is_discontinuous
    )
    print(
        f"Converted {len(documents)} documents with {entity_count} entities "
        f"({discontinuous_count} discontinuous)."
    )
    if "json" in formats:
        print(f"JSON output: {json_dir}")
    if "conll" in formats:
        print(f"CoNLL output: {conll_file}")
        print(f"CoNLL projection conflicts: {conflict_count}")
        for example in conflict_examples:
            print(f"  {example}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
