#!/usr/bin/env python3
"""Convert final_data/raw/<split> into raw_text and BRAT-ready outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple


MARKER_PREFIXES = ("<Related>", "[DONE]", "[CHECK]")
EXCLUDED_ANN_PREFIX = "\tChoose_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="final_data/raw/train",
        help="Directory containing paired raw .txt/.ann files.",
    )
    parser.add_argument(
        "--raw-text-dir",
        default="final_data/raw_text/train",
        help="Directory for trimmed text-only outputs.",
    )
    parser.add_argument(
        "--brat-dir",
        default="final_data/fine/brat/train",
        help="Directory for trimmed BRAT .txt/.ann outputs.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for input and output files.",
    )
    return parser.parse_args()


def strip_footer_markers(text: str) -> str:
    kept_lines: List[str] = []
    for line in text.splitlines():
        if line.startswith(MARKER_PREFIXES):
            break
        kept_lines.append(line)
    return "\n".join(kept_lines).rstrip("\n")


def filter_annotation_lines(lines: Iterable[str]) -> List[str]:
    kept: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if EXCLUDED_ANN_PREFIX in line:
            continue
        kept.append(line)
    return kept


def load_pairs(input_dir: Path) -> List[Tuple[Path, Path]]:
    txt_paths = sorted(input_dir.glob("*.txt"))
    if not txt_paths:
        raise SystemExit(f"No .txt files found in {input_dir}")

    pairs: List[Tuple[Path, Path]] = []
    for txt_path in txt_paths:
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            raise SystemExit(f"Missing paired annotation file for {txt_path}: {ann_path}")
        pairs.append((txt_path, ann_path))
    return pairs


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    raw_text_dir = Path(args.raw_text_dir)
    brat_dir = Path(args.brat_dir)

    raw_text_dir.mkdir(parents=True, exist_ok=True)
    brat_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(input_dir)
    ann_count = 0
    for txt_path, ann_path in pairs:
        trimmed_text = strip_footer_markers(txt_path.read_text(encoding=args.encoding))
        filtered_ann = filter_annotation_lines(ann_path.read_text(encoding=args.encoding).splitlines())

        raw_text_path = raw_text_dir / txt_path.name
        brat_text_path = brat_dir / txt_path.name
        brat_ann_path = brat_dir / ann_path.name

        raw_text_path.write_text(trimmed_text, encoding=args.encoding)
        brat_text_path.write_text(trimmed_text, encoding=args.encoding)
        brat_ann_path.write_text(
            ("\n".join(filtered_ann) + "\n") if filtered_ann else "",
            encoding=args.encoding,
        )
        ann_count += len(filtered_ann)

    print(f"Prepared {len(pairs)} documents from {input_dir}")
    print(f"raw_text output: {raw_text_dir}")
    print(f"BRAT output: {brat_dir}")
    print(f"Kept {ann_count} non-marker annotations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
