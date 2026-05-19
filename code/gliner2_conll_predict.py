#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def normalize_label_list(labels: List[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for raw in labels:
        label = str(raw).strip()
        if not label or label == "O":
            continue
        if "-" in label:
            prefix, rest = label.split("-", 1)
            if prefix in {"B", "I", "S", "E"} and rest:
                label = rest
        if label not in seen:
            seen.add(label)
            normalized.append(label)
    return normalized


def normalize_label_input_mode(label_input_mode: str) -> str:
    mode = str(label_input_mode).strip().lower()
    if mode not in {"short", "def"}:
        raise ValueError("Unsupported label_input_mode {!r}. Expected 'short' or 'def'.".format(label_input_mode))
    return mode


def read_schema_labels(schema_file: str, label_input_mode: str = "def") -> Tuple[List[str], Dict[str, str]]:
    label_input_mode = normalize_label_input_mode(label_input_mode)
    data = json.loads(Path(schema_file).read_text(encoding="utf-8"))
    labels: List[str] = []
    defs: Dict[str, str] = {}

    for entity_type in data.get("entity_types", []):
        coarse_type = str(entity_type.get("coarse_type", "")).strip()
        coarse_def = str(entity_type.get("definition", "")).strip()
        examples = entity_type.get("examples", [])
        example_text = ""
        if isinstance(examples, list) and examples:
            example_text = " Examples: " + "; ".join(str(x) for x in examples)

        fine_types = entity_type.get("fine_types", [])
        if not fine_types and coarse_type:
            if label_input_mode == "def":
                defs[coarse_type] = (coarse_def or coarse_type) + example_text
            labels.append(coarse_type)
            continue

        for fine in fine_types:
            fine_type = str(fine.get("fine_type", "")).strip()
            if not fine_type:
                continue
            fine_def = str(fine.get("definition", "")).strip() or coarse_def or fine_type
            fine_examples = fine.get("examples", [])
            fine_example_text = ""
            if isinstance(fine_examples, list) and fine_examples:
                fine_example_text = " Examples: " + "; ".join(str(x) for x in fine_examples)
            if label_input_mode == "def":
                full_def = fine_def
                if coarse_type and fine_type != coarse_type:
                    full_def = "{} [Category: {}]".format(fine_def, coarse_type)
                defs[fine_type] = full_def + (fine_example_text or example_text)
            labels.append(fine_type)

    return normalize_label_list(labels), defs


def read_labels(args) -> List[str]:
    if args.labels and args.labels_file:
        print("WARNING: Both --labels and --labels-file provided; using --labels-file.", file=sys.stderr)
    if args.labels_file:
        with open(args.labels_file, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        try:
            data = json.loads(txt)
            if isinstance(data, list):
                return normalize_label_list([str(x) for x in data])
        except Exception:
            pass
        return normalize_label_list([line.strip() for line in txt.splitlines() if line.strip()])
    if args.labels:
        return normalize_label_list([x.strip() for x in args.labels.split(",") if x.strip()])
    raise SystemExit("ERROR: You must provide labels via --labels, --labels-file, or --schema-file")


def read_label_definitions(args, labels: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    def load_file(p: str) -> Dict[str, str]:
        txt = open(p, "r", encoding="utf-8").read().strip()
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items()}
        except Exception:
            pass
        result = {}
        for line in txt.splitlines():
            if not line.strip():
                continue
            if "\t" in line:
                k, v = line.split("\t", 1)
                result[k.strip()] = v.strip()
        if not result:
            raise SystemExit(
                "ERROR: --label-defs-file must be JSON {SHORT: Definition} or TSV 'SHORT<TAB>Definition'."
            )
        return result

    if args.label_defs_file:
        mapping = load_file(args.label_defs_file)
    elif args.label_defs:
        pieces = [p for p in args.label_defs.split(";") if p.strip()]
        for p in pieces:
            if "=" not in p:
                raise SystemExit(f"ERROR in --label-defs piece '{p}'. Use SHORT=Definition;...")
            k, v = p.split("=", 1)
            mapping[k.strip()] = v.strip()

    if mapping:
        return {lab: mapping.get(lab, lab) for lab in labels}
    return {}


def read_conll(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    sents_tokens: List[List[str]] = []
    sents_gold: List[List[str]] = []
    cur_tok, cur_gold = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if cur_tok:
                    sents_tokens.append(cur_tok)
                    sents_gold.append(cur_gold)
                    cur_tok, cur_gold = [], []
                continue
            parts = line.split()
            tok = parts[0]
            gold = parts[-1] if len(parts) >= 2 else ""
            cur_tok.append(tok)
            cur_gold.append(gold)
    if cur_tok:
        sents_tokens.append(cur_tok)
        sents_gold.append(cur_gold)

    any_gold = any(any(lbl for lbl in sent) for sent in sents_gold)
    if not any_gold:
        sents_gold = [[] for _ in sents_tokens]
    return sents_tokens, sents_gold


def should_skip_text_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped == "[DONE]":
        return True
    if stripped.startswith("<Related>"):
        return True
    return False


def tokenize_plain_text(line: str) -> List[str]:
    return re.findall(r"\S+", line)


def read_plain_text(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    sents_tokens: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            if should_skip_text_line(raw_line):
                continue
            tokens = tokenize_plain_text(raw_line.strip())
            if tokens:
                sents_tokens.append(tokens)
    return sents_tokens, [[] for _ in sents_tokens]


def tokens_with_char_spans(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    spans: List[Tuple[int, int]] = []
    text_parts: List[str] = []
    offset = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            text_parts.append(" ")
            offset += 1
        start = offset
        end = start + len(tok)
        spans.append((start, end))
        text_parts.append(tok)
        offset = end
    return "".join(text_parts), spans


def gliner2_entities_to_span_dicts(tokens: List[str], token_spans: List[Tuple[int, int]], result: Dict[str, Any]) -> List[Dict[str, Any]]:
    ents: List[Dict[str, Any]] = []
    entities = result.get("entities", {})

    for label, values in entities.items():
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict):
                start = item.get("start")
                end = item.get("end")
                text = item.get("text")
                confidence = item.get("confidence")
                if isinstance(start, int) and isinstance(end, int) and start < end:
                    ent: Dict[str, Any] = {
                        "label": label,
                        "start": start,
                        "end": end,
                        "text": text if isinstance(text, str) else None,
                    }
                    if isinstance(confidence, (int, float)):
                        ent["score"] = float(confidence)
                    ents.append(ent)
                    continue
                if isinstance(text, str):
                    item = text
                else:
                    continue

            if not isinstance(item, str) or not item:
                continue
            surf_tokens = item.split()
            m = len(surf_tokens)
            if m == 0:
                continue
            i = 0
            while i <= len(tokens) - m:
                if tokens[i:i + m] == surf_tokens:
                    start = token_spans[i][0]
                    end = token_spans[i + m - 1][1]
                    ents.append({"label": label, "start": start, "end": end, "text": item})
                    i += m
                else:
                    i += 1
    return ents


def assign_bio(tokens: List[str], token_spans: List[Tuple[int, int]], entities: List[Dict[str, Any]]) -> List[str]:
    n = len(tokens)
    bio = ["O"] * n

    def span_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    best = [None] * n
    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        label = ent["label"]
        span_len = end - start
        for idx, (s, e) in enumerate(token_spans):
            if span_overlap((s, e), (start, end)):
                cur = best[idx]
                cand = (span_len, label)
                if cur is None or cand[0] > cur[0]:
                    best[idx] = cand

    i = 0
    while i < n:
        if best[i] is None:
            i += 1
            continue
        _, lab = best[i]
        bio[i] = f"B-{lab}"
        j = i + 1
        while j < n and best[j] is not None and best[j][1] == lab:
            bio[j] = f"I-{lab}"
            j += 1
        i = j
    return bio


def normalize_entities_for_json(entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ent in entities:
        start = ent.get("start")
        end = ent.get("end")
        label = ent.get("label") or ent.get("type")
        if not isinstance(start, int) or not isinstance(end, int) or start >= end or not label:
            continue
        row: Dict[str, Any] = {
            "start": start,
            "end": end,
            "text": ent.get("text") if isinstance(ent.get("text"), str) else text[start:end],
            "label": str(label),
        }
        score = ent.get("score")
        if isinstance(score, (int, float)):
            row["score"] = float(score)
            row["confidence"] = float(score)
        rows.append(row)
    rows.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict BIO tags with GLiNER2 from a CoNLL file or plain text file.")
    ap.add_argument("--input", default=None, help="Input CoNLL file (token [gold], blank line between sentences).")
    ap.add_argument("--text-input", default=None, help="Plain text input file. Each non-empty line is treated as one sentence.")
    ap.add_argument("--output", required=True, help="Output file with predicted BIO tags.")
    ap.add_argument("--json-output", default=None, help="Optional JSON output path for candidate entity spans.")
    ap.add_argument("--schema-file", default=None, help="Entity schema JSON. If provided, labels and definitions are derived automatically.")
    ap.add_argument("--label-input-mode", choices=["short", "def"], default="def", help="Whether schema-derived GLiNER labels are passed as names only or with definitions.")
    ap.add_argument("--labels", default=None, help="Comma-separated labels, e.g. 'TASK,METRIC,DATASET'.")
    ap.add_argument("--labels-file", default=None, help="File with labels (one per line or JSON list).")
    ap.add_argument("--label-defs", default=None, help="Inline SHORT=Definition entries separated by ';'.")
    ap.add_argument("--label-defs-file", default=None, help="File with label definitions: JSON {SHORT: Definition} or TSV.")
    ap.add_argument("--model", default="fastino/gliner2-base-v1", help="GLiNER2 model name (Hugging Face id).")
    ap.add_argument("--threshold", type=float, default=None, help="Optional confidence threshold (if supported).")
    ap.add_argument("--include-gold", action="store_true", help="If input had gold labels, include them as a middle column.")
    args = ap.parse_args()

    if bool(args.input) == bool(args.text_input):
        raise SystemExit("ERROR: Provide exactly one of --input or --text-input")

    if args.schema_file:
        labels, label_defs = read_schema_labels(args.schema_file, args.label_input_mode)
    else:
        labels = read_labels(args)
        label_defs = read_label_definitions(args, labels)

    label_spec: Any = label_defs if label_defs else labels

    print(f"[INFO] Using labels: {labels}", file=sys.stderr)
    print(f"[INFO] Using label definitions: {label_defs}", file=sys.stderr)
    print(f"[INFO] Using label input mode: {args.label_input_mode}", file=sys.stderr)
    print(f"[INFO] Using model: {args.model}", file=sys.stderr)
    print(f"[INFO] Using label spec: {label_spec}", file=sys.stderr)

    try:
        from gliner2 import GLiNER2
    except ImportError:
        raise SystemExit("ERROR: gliner2 not installed. Run: pip install gliner2")

    print(f"[INFO] Loading GLiNER2 model: {args.model}", file=sys.stderr)
    extractor = GLiNER2.from_pretrained(args.model)

    if args.text_input:
        sents_tokens, sents_gold = read_plain_text(args.text_input)
    else:
        sents_tokens, sents_gold = read_conll(args.input)

    json_rows: List[Dict[str, Any]] = []
    with open(args.output, "w", encoding="utf-8") as out:
        for sid, (tokens, gold) in enumerate(zip(sents_tokens, sents_gold)):
            if not tokens:
                out.write("\n")
                continue
            text, tok_spans = tokens_with_char_spans(tokens)

            try:
                if args.threshold is not None:
                    result = extractor.extract_entities(text, label_spec, threshold=args.threshold, include_confidence=True, include_spans=True)
                else:
                    result = extractor.extract_entities(text, label_spec, include_confidence=True, include_spans=True)
            except TypeError:
                try:
                    if args.threshold is not None:
                        result = extractor.extract_entities(text, label_spec, threshold=args.threshold)
                    else:
                        result = extractor.extract_entities(text, label_spec)
                except TypeError:
                    result = extractor.extract_entities(text, label_spec)

            ent_spans = gliner2_entities_to_span_dicts(tokens, tok_spans, result)
            if args.json_output:
                json_rows.append({
                    "sentence_id": sid,
                    "text": text,
                    "tokens": tokens,
                    "entities": normalize_entities_for_json(ent_spans, text),
                })
            pred_bio = assign_bio(tokens, tok_spans, ent_spans)

            if sents_gold and gold and args.include_gold:
                for t, g, p in zip(tokens, gold, pred_bio):
                    out.write(f"{t}\t{g}\t{p}\n")
            else:
                for t, p in zip(tokens, pred_bio):
                    out.write(f"{t}\t{p}\n")
            out.write("\n")

    print(f"[DONE] Wrote predictions to: {args.output}", file=sys.stderr)
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(json_rows, f, ensure_ascii=True, indent=2)
        print(f"[DONE] Wrote JSON candidates to: {args.json_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
