# PrionNER Release Package

This repository accompanies the paper "PrionNER: A Named Entity Recognition Dataset for Prion Disease Biomedical Literature" and provides the public release of `PrionNER`, a named entity recognition dataset for prion disease biomedical literature. The release includes canonical train/test splits, fine-grained and coarse-grained annotations, synchronized BRAT/JSON/CoNLL formats, and baseline data-processing and evaluation scripts.

The Hugging Face dataset release is available at <https://huggingface.co/datasets/dtan/PrionNER>.

The package contains:

- the canonical `train` and `test` splits
- two annotation granularities: `fine` and `coarse`
- multiple synchronized dataset views: raw BRAT, text-only, JSON, and CoNLL
- minimal code for conversion, evaluation, and baseline training
- metadata and supporting documentation

## Package At A Glance

- Documents: 317 total
- Train split: 247 documents
- Test split: 70 documents
- Fine-grained schema: 31 defined labels, 30 observed in the released data
- Coarse-grained schema: 15 labels
- Fine/coarse entity count: 4,655 train and 1,650 test
- Discontinuous entities: 97 train and 34 test

The fine-grained schema defines `VPSPr`, but that label does not appear in the current released splits.

## Directory Structure

```text
final_submission_PrionNER/
├── data/
│   ├── raw_text/
│   │   ├── train/              # text-only copies, one .txt per document
│   │   └── test/
│   ├── fine/
│   │   ├── brat/               # fine-grained BRAT annotations
│   │   │   ├── train/
│   │   │   └── test/
│   │   ├── json/               # one JSON file per document
│   │   │   ├── train/
│   │   │   └── test/
│   │   └── conll/              # token-level BIO export
│   │       ├── train.conll
│   │       └── test.conll
│   └── coarse/
│       ├── brat/               # coarse-grained BRAT annotations
│       │   ├── train/
│       │   └── test/
│       ├── json/               # one JSON file per document
│       │   ├── train/
│       │   └── test/
│       └── conll/              # token-level BIO export
│           ├── train.conll
│           └── test.conll
├── metadata/                   # schema files, BRAT config, machine-readable summary
├── code/                       # conversion, evaluation, and baseline scripts
├── docs/                       # dataset card, reproducibility notes, figures, notes
├── LICENSE
├── DATA_LICENSE.md
├── THIRD_PARTY_RIGHTS.md
└── README.md
```

## What Each Data Folder Means

### `data/raw_text/`

This contains text-only document files, one `.txt` per document.

- `train/`: 247 files
- `test/`: 70 files

Use this when you need only the document text without BRAT annotations.

### `data/fine/`

This is the fine-grained annotation release. The same split is provided in three synchronized formats:

- `brat/`: original BRAT annotation pairs with fine-grained labels
- `json/`: one JSON file per document, including full text and entity spans
- `conll/`: BIO-tagged token classification export

Use `fine/` if you want the most specific label set.

### `data/coarse/`

This mirrors `data/fine/`, but with labels collapsed into a smaller coarse schema.

- `brat/`: coarse BRAT annotations
- `json/`: coarse JSON export
- `conll/`: coarse BIO-tagged export

Use `coarse/` when you want a simpler label space or a lower-granularity modeling target.

## File Formats

### BRAT files

BRAT files live under `data/fine/brat/` and `data/coarse/brat/`.

- `.txt` stores the document text
- `.ann` stores text-bound annotations
- discontinuous spans use BRAT's semicolon-separated offset format, for example `554 565;588 595`

Example:

- [data/fine/brat/train/prion_0744.ann](data/fine/brat/train/prion_0744.ann)

### JSON files

JSON files live under `data/fine/json/` and `data/coarse/json/`, with one file per document.

Each JSON document has this structure:

- `doc_id`: document identifier
- `text`: full document text
- `entities`: list of entity objects

Each entity object includes:

- `id`: original BRAT entity ID such as `T11`
- `label`: entity type
- `start`, `end`: document-level character offsets
- `text`: entity surface text
- `spans`: one or more `{start, end}` span objects
- `is_discontinuous`: present when the entity spans multiple non-contiguous segments

The `spans` array is important because the dataset includes discontinuous annotations.

Examples:

- [data/fine/json/train/prion_0000.json](data/fine/json/train/prion_0000.json)
- [data/fine/json/train/prion_0744.json](data/fine/json/train/prion_0744.json)

### CoNLL files

CoNLL files live under:

- [data/fine/conll/train.conll](data/fine/conll/train.conll)
- [data/fine/conll/test.conll](data/fine/conll/test.conll)
- [data/coarse/conll/train.conll](data/coarse/conll/train.conll)
- [data/coarse/conll/test.conll](data/coarse/conll/test.conll)

Format:

- one token per line
- `TOKEN<TAB>TAG`
- blank line between text lines/sentences
- BIO tags such as `B-Symptom`, `I-Generic_Prion`, and `O`

This format is convenient for token-classification baselines. The richer BRAT/JSON views are the better source if you need exact discontinuous structure.

## Recommended Entry Points

Choose the folder based on your goal:

- model-ready document objects with character offsets: `data/fine/json/` or `data/coarse/json/`
- token-classification baselines: `data/fine/conll/` or `data/coarse/conll/`
- simplest text-only corpus access: `data/raw_text/`

## Metadata

The main metadata files are:

- [metadata/dataset_summary.json](metadata/dataset_summary.json): machine-readable dataset counts
- [metadata/prion_ner_entity_definitions_fine.json](metadata/prion_ner_entity_definitions_fine.json): fine-grained label definitions and examples
- [metadata/prion_ner_entity_definitions_coarse.json](metadata/prion_ner_entity_definitions_coarse.json): coarse label definitions and examples
- [metadata/annotation.conf](metadata/annotation.conf): BRAT UI configuration

## Code And Docs

- [code/](code): conversion, evaluation, export, and training scripts
- [docs/dataset_card.md](docs/dataset_card.md): compact dataset overview
- [docs/reproducibility.md](docs/reproducibility.md): commands for rebuilding exports and running baselines
- [docs/release_note_licensing.md](docs/release_note_licensing.md): short public-facing note explaining the split licensing policy
- [docs/figures](docs/figures): examples of discontinuous, nested, and overlapping structures

## Quick Start

From this directory:

```bash
python3 code/brat_to_json_conll.py --input-dir data/fine/brat/train --formats both
python3 code/evaluate_conll_predictions.py \
  --gold-file data/fine/conll/test.conll \
  --pred-file YOUR_PREDICTIONS.conll \
  --output-dir eval_out \
  --model-name your_model
```

## Licensing and Redistribution

This release uses a split licensing policy.

- Code in `code/` is licensed under the [MIT License](LICENSE).
- Project-authored documentation, metadata, annotations, and derived structured outputs are intended to be released under [CC BY 4.0](DATA_LICENSE.md), to the extent the maintainers hold the necessary rights.
- Underlying article or abstract text will remain in this public release, but it is **not** blanket-relicensed by the project. See [THIRD_PARTY_RIGHTS.md](THIRD_PARTY_RIGHTS.md).

If you want the safest reuse path, prefer using the annotation layer, schema files, statistics, and project-authored metadata rather than redistributing the underlying source text.

## Release Notes

- Trained weights, large experiment artifacts, caches, and internal workspaces are intentionally excluded from this package.
- The source-text notice in [THIRD_PARTY_RIGHTS.md](THIRD_PARTY_RIGHTS.md) should be preserved in any redistributed copy of this package.
