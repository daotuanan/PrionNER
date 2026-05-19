#!/usr/bin/env python3
"""Train and evaluate a supervised GLiNER2 NER model on CoNLL BIO data."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from train_bert_ner import (
    bio_to_entities,
    compute_metrics_from_tags,
    read_conll,
    set_seed,
    write_per_label_tsv,
    write_predictions,
)

import gliner2_conll_predict


ROOT_DIR = Path(__file__).resolve().parent
MODEL_ALIASES = {
    "gliner2_base": "fastino/gliner2-base-v1",
    "gliner2_large": "fastino/gliner2-large-v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="final_data/fine/conll/train.conll")
    parser.add_argument("--test-file", default="final_data/fine/conll/test.conll")
    parser.add_argument("--schema-file", default=None, help="Entity schema JSON for label descriptions.")
    parser.add_argument(
        "--label-input-mode",
        choices=["short", "def"],
        default="def",
        help="Use schema label names only or schema-derived label definitions.",
    )
    parser.add_argument("--model-name", required=True, help="GLiNER2 HF model ID or alias.")
    parser.add_argument("--output-dir", required=True, help="Directory for model and evaluation outputs.")
    parser.add_argument("--num-train-epochs", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--task-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--scheduler-type", default="cosine", choices=["linear", "cosine", "cosine_restarts", "constant"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--eval-strategy", default="epoch", choices=["epoch", "steps", "no"])
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None, help="Optional inference threshold for test-set decoding.")
    parser.add_argument("--use-lora", action="store_true", help="Enable parameter-efficient LoRA fine-tuning.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--save-adapter-only", action="store_true", help="When using LoRA, save checkpoints as adapters only.")
    return parser.parse_args()


def resolve_model_name(raw: str) -> str:
    return MODEL_ALIASES.get(raw.lower(), raw)


def read_label_descriptions(schema_file: Optional[str], label_input_mode: str) -> Dict[str, str]:
    if not schema_file or label_input_mode != "def":
        return {}
    _, label_defs = gliner2_conll_predict.read_schema_labels(schema_file, label_input_mode="def")
    return label_defs


def sentence_entities_from_bio(tokens: Sequence[str], tags: Sequence[str]) -> Dict[str, List[str]]:
    entities_by_label: Dict[str, List[str]] = defaultdict(list)
    for label, start, end in bio_to_entities(tags):
        mention = " ".join(tokens[start:end]).strip()
        if mention:
            entities_by_label[label].append(mention)
    return dict(entities_by_label)


def build_input_examples(
    sentences_tokens: Sequence[Sequence[str]],
    sentences_tags: Sequence[Sequence[str]],
    label_descriptions: Dict[str, str],
):
    try:
        from gliner2.training.data import InputExample
    except ImportError as exc:
        raise SystemExit("ERROR: gliner2 not installed in the selected environment. Use .venv/bin/python.") from exc

    examples = []
    kept_count = 0
    dropped_empty_count = 0

    for tokens, tags in zip(sentences_tokens, sentences_tags):
        text = " ".join(tokens).strip()
        if not text:
            continue

        entities = sentence_entities_from_bio(tokens, tags)
        if not entities:
            dropped_empty_count += 1
            continue

        entity_descriptions = {label: label_descriptions[label] for label in entities if label in label_descriptions}
        examples.append(
            InputExample(
                text=text,
                entities=entities,
                entity_descriptions=entity_descriptions or None,
            )
        )
        kept_count += 1

    return examples, {"kept": kept_count, "dropped_empty": dropped_empty_count, "total": len(sentences_tokens)}


def run_gliner2_prediction(model, sentences_tokens: Sequence[Sequence[str]], label_spec, threshold: Optional[float]) -> List[List[str]]:
    pred_sequences: List[List[str]] = []
    for tokens in sentences_tokens:
        if not tokens:
            pred_sequences.append([])
            continue
        text, tok_spans = gliner2_conll_predict.tokens_with_char_spans(list(tokens))
        try:
            if threshold is None:
                result = model.extract_entities(text, label_spec, include_confidence=True, include_spans=True)
            else:
                result = model.extract_entities(
                    text,
                    label_spec,
                    threshold=threshold,
                    include_confidence=True,
                    include_spans=True,
                )
        except TypeError:
            if threshold is None:
                result = model.extract_entities(text, label_spec)
            else:
                result = model.extract_entities(text, label_spec, threshold=threshold)
        entity_spans = gliner2_conll_predict.gliner2_entities_to_span_dicts(list(tokens), tok_spans, result)
        pred_sequences.append(gliner2_conll_predict.assign_bio(list(tokens), tok_spans, entity_spans))
    return pred_sequences


def load_gliner2_model(model_name_or_path: str):
    try:
        from gliner2 import GLiNER2
    except ImportError as exc:
        raise SystemExit("ERROR: gliner2 not installed in the selected environment. Use .venv/bin/python.") from exc
    return GLiNER2.from_pretrained(model_name_or_path)


def load_evaluation_model(base_model_name: str, checkpoint_dir: Path, adapter_only: bool):
    if adapter_only:
        model = load_gliner2_model(base_model_name)
        model.load_adapter(str(checkpoint_dir))
        return model
    return load_gliner2_model(str(checkpoint_dir))


def print_runtime_debug() -> None:
    try:
        import torch
    except ImportError:
        print("[DEBUG] torch import failed", flush=True)
        return

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"[DEBUG] CUDA_VISIBLE_DEVICES={cuda_visible_devices}", flush=True)
    print(f"[DEBUG] torch_version={torch.__version__}", flush=True)
    print(f"[DEBUG] torch_cuda_is_available={torch.cuda.is_available()}", flush=True)
    print(f"[DEBUG] torch_cuda_device_count={torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"[DEBUG] torch_cuda_device_{index}={torch.cuda.get_device_name(index)}", flush=True)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    print_runtime_debug()

    train_path = Path(args.train_file)
    test_path = Path(args.test_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tokens, train_tags = read_conll(train_path)
    test_tokens, test_tags = read_conll(test_path)
    label_descriptions = read_label_descriptions(args.schema_file, args.label_input_mode)
    label_names = sorted({tag[2:] for seq in (train_tags + test_tags) for tag in seq if tag != "O"})
    if args.label_input_mode == "def":
        label_spec = {label: label_descriptions[label] for label in label_names if label in label_descriptions} or label_names
    else:
        label_spec = label_names

    train_examples, train_stats = build_input_examples(train_tokens, train_tags, label_descriptions)
    eval_examples, eval_stats = build_input_examples(test_tokens, test_tags, label_descriptions)
    if not train_examples:
        raise SystemExit("ERROR: No non-empty entity training examples were found after CoNLL conversion.")

    model_name = resolve_model_name(args.model_name)
    model = load_gliner2_model(model_name)

    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    training_config = TrainingConfig(
        output_dir=str(output_dir),
        experiment_name=Path(args.output_dir).name,
        num_epochs=args.num_train_epochs,
        batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        encoder_lr=args.encoder_learning_rate,
        task_lr=args.task_learning_rate,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        save_best=bool(eval_examples),
        logging_steps=args.logging_steps,
        early_stopping=False,
        num_workers=args.num_workers,
        seed=args.seed,
        validate_data=True,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        save_adapter_only=args.save_adapter_only,
    )

    trainer = GLiNER2Trainer(model, training_config)
    print(f"[DEBUG] trainer_device={trainer.device}", flush=True)
    train_summary = trainer.train(train_data=train_examples, eval_data=eval_examples or None)

    checkpoint_name = "best" if (output_dir / "best").exists() else "final"
    checkpoint_dir = output_dir / checkpoint_name
    eval_model = load_evaluation_model(
        base_model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        adapter_only=bool(args.use_lora and args.save_adapter_only),
    )

    pred_sequences = run_gliner2_prediction(eval_model, test_tokens, label_spec, args.threshold)
    test_metrics = compute_metrics_from_tags(test_tags, pred_sequences)

    dataset_summary = {
        "train_examples": train_stats,
        "eval_examples": eval_stats,
        "test_sentences_total": len(test_tokens),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(train_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics_payload = {
        "config": {
            "model_name": model_name,
            "train_file": str(train_path.resolve()),
            "test_file": str(test_path.resolve()),
            "schema_file": "" if args.schema_file is None else str(Path(args.schema_file).resolve()),
            "output_dir": str(output_dir.resolve()),
            "num_train_epochs": args.num_train_epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "encoder_learning_rate": args.encoder_learning_rate,
            "task_learning_rate": args.task_learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "scheduler_type": args.scheduler_type,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed,
            "threshold": args.threshold,
            "use_lora": args.use_lora,
            "save_adapter_only": args.save_adapter_only,
            "evaluation_checkpoint": checkpoint_name,
            "label_input_mode": args.label_input_mode,
            "label_spec_mode": "definitions" if isinstance(label_spec, dict) else "labels",
            "label_names": label_names,
        },
        "dataset_summary": dataset_summary,
        "train_summary": train_summary,
        "eval_metrics": {key: value for key, value in test_metrics.items() if key != "per_label"},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_per_label_tsv(output_dir / "per_label_metrics.tsv", test_metrics["per_label"])
    write_predictions(output_dir / "test_predictions.conll", test_tokens, test_tags, pred_sequences)

    summary = {
        "model_name": model_name,
        "evaluation_checkpoint": checkpoint_name,
        "entity_precision": test_metrics["entity_precision"],
        "entity_recall": test_metrics["entity_recall"],
        "entity_f1": test_metrics["entity_f1"],
        "token_accuracy": test_metrics["token_accuracy"],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
