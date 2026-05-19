#!/usr/bin/env python3
"""Run W2NER training without a held-out development set."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import prettytable as pt
import torch
import torch.nn as nn
import transformers
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w2ner-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--predict-only", action="store_true")
    return parser.parse_args()


def load_w2ner_modules(w2ner_root: Path):
    sys.path.insert(0, str(w2ner_root))
    config_module = importlib.import_module("config")
    data_loader_module = importlib.import_module("data_loader")
    model_module = importlib.import_module("model")
    utils_module = importlib.import_module("utils")
    return config_module, data_loader_module, model_module, utils_module


def load_train_test_data(config_obj, data_loader_module, w2ner_root: Path):
    data_dir = w2ner_root / "data" / config_obj.dataset
    train_path = data_dir / "train.json"
    test_path = data_dir / "test.json"

    with train_path.open("r", encoding="utf-8") as handle:
        train_data = json.load(handle)
    with test_path.open("r", encoding="utf-8") as handle:
        test_data = json.load(handle)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config_obj.bert_name, cache_dir="./cache/")

    vocab = data_loader_module.Vocabulary()
    train_ent_num = data_loader_module.fill_vocab(vocab, train_data)
    test_ent_num = data_loader_module.fill_vocab(vocab, test_data)

    table = pt.PrettyTable([config_obj.dataset, "sentences", "entities"])
    table.add_row(["train", len(train_data), train_ent_num])
    table.add_row(["test", len(test_data), test_ent_num])
    config_obj.logger.info("\n{}".format(table))

    config_obj.label_num = len(vocab.label2id)
    config_obj.vocab = vocab

    train_dataset = data_loader_module.RelationDataset(
        *data_loader_module.process_bert(train_data, tokenizer, vocab)
    )
    test_dataset = data_loader_module.RelationDataset(
        *data_loader_module.process_bert(test_data, tokenizer, vocab)
    )
    return (train_dataset, test_dataset), (train_data, test_data)


class Trainer:
    def __init__(self, model, config_obj, updates_total, logger, utils_module):
        self.model = model
        self.config = config_obj
        self.logger = logger
        self.utils = utils_module
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": config_obj.bert_learning_rate,
                "weight_decay": config_obj.weight_decay,
            },
            {
                "params": [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": config_obj.bert_learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": other_params,
                "lr": config_obj.learning_rate,
                "weight_decay": config_obj.weight_decay,
            },
        ]

        self.optimizer = transformers.AdamW(
            params,
            lr=config_obj.learning_rate,
            weight_decay=config_obj.weight_decay,
        )
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config_obj.warm_factor * updates_total,
            num_training_steps=updates_total,
        )

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []

        for data_batch in data_loader:
            data_batch = [data.cuda() for data in data_batch[:-1]]
            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

            outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            grid_mask2d = grid_mask2d.clone()
            loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            loss_list.append(loss.cpu().item())

            outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)
            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(
            label_result.numpy(),
            pred_result.numpy(),
            average="macro",
        )

        table = pt.PrettyTable([f"Train {epoch}", "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        self.logger.info("\n{}".format(table))
        return f1

    def eval(self, epoch, data_loader):
        self.model.eval()
        pred_result = []
        label_result = []
        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        with torch.no_grad():
            for data_batch in data_loader:
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                grid_mask2d = grid_mask2d.clone()
                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, _ = self.utils.decode(
                    outputs.cpu().numpy(),
                    entity_text,
                    sent_length.cpu().numpy(),
                )

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)
                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(
            label_result.numpy(),
            pred_result.numpy(),
            average="macro",
        )
        e_f1, e_p, e_r = self.utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        self.logger.info(
            "TEST Label F1 {}".format(
                f1_score(label_result.numpy(), pred_result.numpy(), average=None)
            )
        )
        table = pt.PrettyTable([f"TEST {epoch}", "F1", "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])
        self.logger.info("\n{}".format(table))
        return e_f1

    def predict(self, epoch, data_loader, data):
        self.model.eval()
        result = []
        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        pred_result = []
        label_result = []
        offset = 0

        with torch.no_grad():
            for data_batch in data_loader:
                sentence_batch = data[offset:offset + self.config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                grid_mask2d = grid_mask2d.clone()
                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = self.utils.decode(
                    outputs.cpu().numpy(),
                    entity_text,
                    sent_length.cpu().numpy(),
                )

                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence_tokens = sentence["sentence"]
                    label_map = {
                        entity_type.lower(): entity_type
                        for entity_type in {raw.get("type") for raw in sentence.get("ner", []) if isinstance(raw, dict)}
                    }
                    instance = {
                        "id": sentence.get("id"),
                        "doc_id": sentence.get("doc_id"),
                        "line_index": sentence.get("line_index"),
                        "text": sentence_tokens,
                        "token_offsets": sentence.get("token_offsets"),
                        "entity": [],
                    }
                    for ent in ent_list:
                        lower_type = self.config.vocab.id_to_label(ent[1])
                        instance["entity"].append(
                            {
                                "index": list(ent[0]),
                                "text": [sentence_tokens[x] for x in ent[0]],
                                "type": label_map.get(lower_type, lower_type),
                                "type_lower": lower_type,
                            }
                        )
                    result.append(instance)

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)
                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
                offset += self.config.batch_size

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(
            label_result.numpy(),
            pred_result.numpy(),
            average="macro",
        )
        e_f1, e_p, e_r = self.utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        self.logger.info(
            "TEST Label F1 {}".format(
                f1_score(label_result.numpy(), pred_result.numpy(), average=None)
            )
        )
        table = pt.PrettyTable([f"TEST {epoch}", "F1", "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])
        self.logger.info("\n{}".format(table))

        with open(self.config.predict_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False)
        return e_f1

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


def main() -> int:
    args = parse_args()
    w2ner_root = Path(args.w2ner_root).resolve()
    config_module, data_loader_module, model_module, utils_module = load_w2ner_modules(w2ner_root)

    config_args = argparse.Namespace(config=args.config, device=args.device)
    config_obj = config_module.Config(config_args)
    Path("log").mkdir(parents=True, exist_ok=True)
    logger = utils_module.get_logger(config_obj.dataset)
    logger.info(config_obj)
    config_obj.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    random.seed(config_obj.seed)
    np.random.seed(config_obj.seed)
    torch.manual_seed(config_obj.seed)
    torch.cuda.manual_seed(config_obj.seed)
    torch.cuda.manual_seed_all(config_obj.seed)

    logger.info("Loading Data")
    datasets, original_data = load_train_test_data(config_obj, data_loader_module, w2ner_root)

    train_loader = DataLoader(
        dataset=datasets[0],
        batch_size=config_obj.batch_size,
        collate_fn=data_loader_module.collate_fn,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=datasets[1],
        batch_size=config_obj.batch_size,
        collate_fn=data_loader_module.collate_fn,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    updates_total = max(1, len(datasets[0]) // config_obj.batch_size) * config_obj.epochs

    logger.info("Building Model")
    model = model_module.Model(config_obj).cuda()
    trainer = Trainer(model, config_obj, updates_total, logger, utils_module)

    last_test_f1 = 0.0
    if args.predict_only:
        logger.info("Loading checkpoint for prediction only: {}".format(config_obj.save_path))
        trainer.load(config_obj.save_path)
        last_test_f1 = trainer.eval("Loaded", test_loader)
    else:
        for epoch in range(config_obj.epochs):
            logger.info("Epoch: {}".format(epoch))
            trainer.train(epoch, train_loader)
            last_test_f1 = trainer.eval(epoch, test_loader)

        trainer.save(config_obj.save_path)
        logger.info("Final TEST F1: {:3.4f}".format(last_test_f1))
    trainer.predict("Final", test_loader, original_data[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
