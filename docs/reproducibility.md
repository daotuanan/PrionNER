# Reproducibility Notes

## Included Scripts

The release package includes only the scripts needed to work with the released dataset directly:

- `code/prepare_raw_split.py`: trims raw `.txt` files and filters marker annotations
- `code/brat_to_json_conll.py`: converts BRAT annotations to JSON and CoNLL BIO
- `code/build_coarse_dataset.py`: derives coarse BRAT annotations from the fine BRAT set
- `code/export_w2ner_dataset.py`: exports BRAT data to the JSON format expected by W2NER
- `code/convert_w2ner_output_to_brat.py`: converts W2NER predictions back to BRAT
- `code/evaluate_conll_predictions.py`: computes token and exact-entity metrics on CoNLL BIO predictions
- `code/train_bert_ner.py`: trains a BERT-family token classification baseline
- `code/train_gliner2_ner.py`: trains and evaluates a GLiNER2 baseline
- `code/gliner2_conll_predict.py`: GLiNER2 helper utilities used by the GLiNER2 baseline
- `code/run_w2ner_no_dev.py`: helper runner for W2NER training or prediction when the upstream W2NER repo is available locally

## Python Dependencies

Install the common dependencies from [requirements-public.txt](../code/requirements-public.txt). `gliner2` is only needed for the GLiNER2 baseline.

## Common Workflows

Rebuild JSON and CoNLL from BRAT:

```bash
python3 code/brat_to_json_conll.py --input-dir data/fine/brat/train --formats both
python3 code/brat_to_json_conll.py --input-dir data/fine/brat/test --formats both
python3 code/brat_to_json_conll.py --input-dir data/coarse/brat/train --formats both
python3 code/brat_to_json_conll.py --input-dir data/coarse/brat/test --formats both
```

Rebuild coarse BRAT from fine BRAT:

```bash
python3 code/build_coarse_dataset.py \
  --fine-brat-dir data/fine/brat/train \
  --coarse-brat-dir data/coarse/brat/train \
  --schema-file metadata/prion_ner_entity_definitions_fine.json
python3 code/build_coarse_dataset.py \
  --fine-brat-dir data/fine/brat/test \
  --coarse-brat-dir data/coarse/brat/test \
  --schema-file metadata/prion_ner_entity_definitions_fine.json
```

Train a BERT-family baseline:

```bash
python3 code/train_bert_ner.py \
  --train-file data/fine/conll/train.conll \
  --test-file data/fine/conll/test.conll \
  --model-name dmis-lab/biobert-base-cased-v1.2 \
  --output-dir bert_runs/biobert_fine
```

Train a GLiNER2 baseline:

```bash
python3 code/train_gliner2_ner.py \
  --train-file data/fine/conll/train.conll \
  --test-file data/fine/conll/test.conll \
  --schema-file metadata/prion_ner_entity_definitions_fine.json \
  --model-name fastino/gliner2-large-v1 \
  --output-dir gliner2_runs/fine_def
```

Evaluate CoNLL predictions:

```bash
python3 code/evaluate_conll_predictions.py \
  --gold-file data/fine/conll/test.conll \
  --pred-file YOUR_PREDICTIONS.conll \
  --output-dir eval_out \
  --model-name your_model
```

## W2NER Note

`code/run_w2ner_no_dev.py` requires an external checkout of the upstream W2NER repository and a matching config file. This package includes the export and prediction-conversion helpers, but it does not vendor the W2NER codebase itself.
