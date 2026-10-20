#!/usr/bin/env python3
"""Fine-tune Kev on local CUDA GPUs, then calibrate and compare against the starting model.

Run from a Kev checkout with its dependencies installed, plus PyYAML and optionally wandb:

    uv run --with pyyaml python my_scripts/finetune_local.py --example-config > local.yaml
    # Edit local.yaml, especially data.path and the required full-training settings.
    CUDA_VISIBLE_DEVICES=0,1 uv run --with pyyaml --with wandb python my_scripts/finetune_local.py --config local.yaml

One entry point handles conversion, splitting, distributed training, and evaluation. It imports
Kev's training and scoring helpers and the existing adapter merge utility. No Modal is used.
Paths in the input YAML are relative to the working directory. The output path is
a parent directory; local.yaml writes to <output>/local1, then local2, and so on.

Full mode requires explicit lr, batch, accum, and checkpointing. LoRA mode inherits these from
the starting adapter unless overridden, with its default lr capped at 5e-5. Batch is per GPU;
effective batch = batch * accum * visible GPU count. Both modes use all CUDA-visible GPUs.
Periodic evaluation borrows the live model on every rank, without an extra model allocation.
It scores the development set at T=1 every 10 optimizer steps; calibration happens at the end.
The final checkpoint is the last training step, with no early stopping or best-model selection.
Final and baseline evaluation fall back to CPU if a checkpoint cannot be scored on one GPU.

data.path accepts native Kev JSONL or a directory containing train/calibration/development.jsonl.
For conversion, set data.spec to a workload JSON, data.state to input columns, data.labels to a
question-to-column mapping, and data.mappings to question-to-label-renaming dictionaries.
data.holdout accepts native Kev JSONL reserved for calibration/development. Conversion applies
only to data.path; convert a tabular holdout to Kev JSONL before supplying it.

The output contains config.yaml with all resolved settings, prepared data and its summary,
train.log, metrics.jsonl, periodic reports, final and baseline reports, errors.jsonl, result.json,
and checkpoint/. A merged full-weight initialization, when needed, is kept in merged-init/.
W&B is off by default. When enabled only rank 0's metrics are logged, through the parent process.
API keys belong in the environment, never in YAML. Reports can contain private input text.

Based on skills/kev-finetune/scripts/{kev_modal,convert_data,split_data}.py.
"""
import argparse
import copy
import csv
import gc
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PARTITIONS = ("train", "calibration", "development")
EVENT = "KEV_METRIC "
DEFAULTS = {
    "init_from": "jaredpalmer/kev-4b", "output": None, "mode": "full",
    "data": {"path": None, "spec": None, "state": [], "labels": {}, "mappings": {},
             "score_offset": 0, "holdout": None, "calibration_fraction": 0.15, "development_fraction": 0.15},
    "train": {"epochs": 1, "seed": 0, "lr": None, "batch": None, "accum": None, "checkpointing": None,
              "head_lr": 0.0, "weight_decay": 0.01, "max_state": None, "max_steps": 0,
              "length_sort": False, "pass_tokens_max": 0, "row_budget": 0, "shared_prefix": None},
    "evaluation": {"every_steps": 10, "errors_every_eval": False, "error_limit": 60,
                   "bootstrap_samples": 1000, "bootstrap_seed": 0},
    "wandb": {"enabled": False, "project": "kev-finetune", "entity": None, "name": None,
              "mode": "online", "tags": []},
}


def yaml_read(path):
    import yaml
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def yaml_write(path, value):
    import yaml
    Path(path).write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def configuration(value):
    """Reject misspelled options before loading a model or creating an output directory."""
    if not isinstance(value, dict): raise ValueError("configuration must be a YAML mapping")
    result = copy.deepcopy(DEFAULTS)
    for key, item in value.items():
        if key not in result: raise ValueError(f"unknown configuration key: {key}")
        if isinstance(result[key], dict):
            if not isinstance(item, dict): raise ValueError(f"{key} must be a mapping")
            unknown = item.keys() - result[key].keys()
            if unknown: raise ValueError(f"unknown {key} settings: {sorted(unknown)}")
            result[key].update(item)
        else:
            result[key] = item
    if result["mode"] not in ("full", "lora"): raise ValueError("mode must be full or lora")
    if not result["output"] or not result["data"]["path"]: raise ValueError("output and data.path are required")
    for key in ("lr", "batch", "accum", "checkpointing"):
        if result["mode"] == "full" and result["train"][key] is None:
            raise ValueError(f"full fine-tuning requires an explicit train.{key}; LoRA settings are not inherited")
    data, evaluation = result["data"], result["evaluation"]
    if not isinstance(data["state"], list) or not all(isinstance(v, str) and v for v in data["state"]):
        raise ValueError("data.state must be a list of column names")
    for key in ("labels", "mappings"):
        if not isinstance(data[key], dict): raise ValueError(f"data.{key} must be a mapping")
    for key, item in result["train"].items():
        if item is not None and (not isinstance(item, (bool, int, float)) or not math.isfinite(item)):
            raise ValueError(f"train.{key} must be a finite number or boolean")
    fractions = [data["calibration_fraction"], data["development_fraction"]]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in fractions) or sum(fractions) >= 1:
        raise ValueError("calibration/development fractions must be positive and sum to less than one")
    for key in ("every_steps", "error_limit", "bootstrap_samples"):
        if type(evaluation[key]) is not int or evaluation[key] < 1: raise ValueError(f"evaluation.{key} must be a positive integer")
    for key, settings in (("enabled", result["wandb"]), ("errors_every_eval", evaluation)):
        if type(settings[key]) is not bool: raise ValueError(f"{key} must be a YAML boolean")
    if result["wandb"]["mode"] not in ("online", "offline", "disabled"): raise ValueError("invalid wandb.mode")
    return result


def next_output_dir(parent, config_name):
    """Name a new run after the config file and the highest existing suffix."""
    parent = Path(parent)
    pattern = re.compile(re.escape(config_name) + "_" + r"(\d+)")
    highest = max(
        (int(match.group(1)) for path in parent.iterdir()
         if (match := pattern.fullmatch(path.name))),
        default=0,
    ) if parent.exists() else 0
    return parent / f"{config_name}_{highest + 1}"


def state_key(state):
    from kev.suite import text_digest
    return text_digest(state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False))


def field(row, column):
    for part in column.split("."):
        if not isinstance(row, dict) or part not in row: return None
        row = row[part]
    return row


def input_rows(path):
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as stream:
        if path.suffix.lower() in (".jsonl", ".ndjson"):
            for line in stream:
                if not line.strip(): continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    yield None, f"invalid JSON: {error.msg}"
                else:
                    yield value, None
        elif path.suffix.lower() == ".json":
            data = json.load(stream)
            for row in data if isinstance(data, list) else data.get("rows", data.get("data", [])):
                yield row, None
        else:
            for row in csv.DictReader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ","):
                yield row, None


def converted_label(question, value, mappings, offset):
    if value is None: raise ValueError("empty label")
    text = mappings.get(str(value).strip(), str(value).strip())
    if not text: raise ValueError("empty label")
    if question["type"] == "noul":
        if str(text).lower() in ("true", "t", "yes", "y", "1", "on"): return True
        if str(text).lower() in ("false", "f", "no", "n", "0", "off"): return False
        raise ValueError("not a yes/no label")
    criteria = question["criteria"]
    if question["type"] == "choice":
        if text in criteria: return text
        hits = [k for k in criteria if k.casefold() == str(text).casefold()]
        hits = hits or [k for k, v in criteria.items() if isinstance(v, str) and v.casefold() == str(text).casefold()]
    else:
        try:
            number = float(text)
        except (ValueError, TypeError):
            hits = [i for i, v in enumerate(criteria) if str(v).casefold() == str(text).casefold()
                    or str(v).casefold().startswith(str(text).casefold() + ":")]
        else:
            if not math.isfinite(number) or not number.is_integer(): raise ValueError("score must be an integer level")
            label = int(number) - offset
            if not 0 <= label < len(criteria): raise ValueError("score is outside the criteria")
            return label
    if len(hits) != 1: raise ValueError("label does not identify a unique option; supply data.mappings")
    return hits[0]


def validate_record(record):
    from kev.api import SystemOneRequest, question_keys
    if not isinstance(record, dict): raise ValueError("record must be an object")
    SystemOneRequest.model_validate(record)
    if record.get("state") in (None, "", {}, []): raise ValueError("empty state")
    for qid, q in record["questions"].items():
        if q.get("instructions") in (None, ""): raise ValueError(f"{qid}: missing instructions")
        keys = question_keys(q["type"], q.get("criteria"))
        label = q.get("label")
        if q["type"] == "noul": valid = type(label) is bool
        elif q["type"] == "score": valid = type(label) is int and 0 <= label < len(keys) and len(keys) >= 2
        else: valid = isinstance(label, str) and label in keys
        if not valid: raise ValueError(f"{qid}: invalid {q['type']} label")
        target = q.get("target")
        if target is not None:
            if not isinstance(target, dict) or target.keys() - set(keys): raise ValueError(f"{qid}: invalid target keys")
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in target.values()) or sum(target.values()) <= 0:
                raise ValueError(f"{qid}: invalid target weights")


def read_data(path, data, convert=False):
    """Skip invalid records with counted reasons; preserve source metadata for training admission."""
    from kev.suite import read_json
    spec = read_json(data["spec"])["questions"] if convert else None
    if convert and not data["state"]: raise ValueError("conversion requires data.state columns")
    if convert:
        unknown = (data["labels"].keys() | data["mappings"].keys()) - spec.keys()
        if unknown: raise ValueError(f"conversion mappings name unknown questions: {sorted(unknown)}")
    records, rejected = [], Counter()
    for row, problem in input_rows(path):
        if problem:
            rejected[problem] += 1
            continue
        try:
            if convert:
                values = {column: field(row, column) for column in data["state"]}
                state = next(iter(values.values())) if len(values) == 1 else {k: v for k, v in values.items() if v not in (None, "")}
                record = {"state": state, "questions": {}}
                for qid, q in spec.items():
                    label = converted_label(q, field(row, data["labels"].get(qid, qid)), data["mappings"].get(qid, {}), data["score_offset"])
                    record["questions"][qid] = {**q, "label": label}
            else:
                record = row
            validate_record(record)
        except (ValueError, TypeError, KeyError) as error:
            rejected[str(error)] += 1
            continue
        records.append(record)
    return records, dict(rejected)


def clean_records(records):
    """Drop exact duplicates and all records belonging to a state with conflicting answers."""
    seen, labels, conflicts, kept = set(), {}, set(), []
    for record in records:
        fingerprint = json.dumps({k: record[k] for k in ("state", "questions")}, sort_keys=True, ensure_ascii=False)
        if fingerprint in seen: continue
        seen.add(fingerprint)
        state = state_key(record["state"])
        for qid, q in record["questions"].items():
            key = state, qid, json.dumps({k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}, sort_keys=True)
            answer = json.dumps({k: q[k] for k in ("label", "target") if k in q}, sort_keys=True)
            if labels.setdefault(key, answer) != answer: conflicts.add(state)
        kept.append(record)
    return [r for r in kept if state_key(r["state"]) not in conflicts], {
        "duplicates_dropped": len(records) - len(kept), "conflicting_states_dropped": len(conflicts)}


def split_records(records, calibration, development, seed):
    groups = defaultdict(list)
    for record in records: groups[state_key(record["state"])].append(record)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_cal, n_dev = round(len(keys) * calibration), round(len(keys) * development)
    slices = (keys[n_cal + n_dev:], keys[n_dev:n_dev + n_cal], keys[:n_dev])
    return {name: [r for key in selected for r in groups[key]] for name, selected in zip(PARTITIONS, slices)}


def prepare_data(config, tokenizer):
    from kev.api import question_keys
    from kev.data import load_records, materialize
    from kev.model import fits, training_context
    from kev.suite import digest, write_json, write_jsonl
    data, out = config["data"], Path(config["output"]) / "data"
    out.mkdir()
    path = Path(data["path"])
    summary = {"inputs": {}, "cleanup": {}, "partitions": {}}

    def read(path, convert=False):
        records, invalid = read_data(path, data, convert)
        records, cleanup = clean_records(records)
        summary["inputs"][str(path)] = digest(path)
        summary["cleanup"][str(path)] = {**cleanup, "invalid": invalid}
        return records

    if path.is_dir():
        if data["spec"] or data["holdout"]: raise ValueError("prepared partitions cannot be combined with data.spec or data.holdout")
        parts = {name: read(path / f"{name}.jsonl") for name in PARTITIONS}
    else:
        records = read(path, bool(data["spec"]))
        if data["spec"]: summary["inputs"][data["spec"]] = digest(data["spec"])
        if data["holdout"]:
            holdout = read(Path(data["holdout"]))
            held_states = {state_key(r["state"]) for r in holdout}
            train = [r for r in records if state_key(r["state"]) not in held_states]
            summary["holdout_overlap_dropped"] = len(records) - len(train)
            parts = split_records(holdout, 0.5, 0.5, config["train"]["seed"])
            parts["calibration"].extend(parts["train"])
            parts["train"] = train
        else:
            parts = split_records(records, data["calibration_fraction"], data["development_fraction"], config["train"]["seed"])
    state_sets = {p: {state_key(r["state"]) for r in rows} for p, rows in parts.items()}
    for i, part in enumerate(PARTITIONS):
        for other in PARTITIONS[i + 1:]:
            if state_sets[part] & state_sets[other]: raise ValueError(f"state overlap between {part} and {other}")
    context = training_context(config["training_args"]["max_state"])
    for name, rows in parts.items():
        if not rows: raise ValueError(f"empty {name} partition after cleanup; supply more records")
        # Stable record IDs and state groups give comparisons and bootstrap a common identity.
        for i, row in enumerate(rows):
            row["_meta"] = {**row.get("_meta", {}), "id": f"{name}/{i}", "group_id": state_key(row["state"]), "variant": "clean"}
        target = out / f"{name}.jsonl"
        write_jsonl(target, rows)
        loaded = load_records(target)
        overlong = [r["_meta"]["id"] for r in loaded if not fits(materialize(r), tokenizer, **context)]
        counts, expected = defaultdict(Counter), defaultdict(set)
        for r in loaded:
            for qid, q in r["questions"].items():
                label = str(q["label"]).lower() if q["type"] == "noul" else str(q["label"])
                counts[qid][label] += 1
                expected[qid].update(question_keys(q["type"], q.get("criteria")))
        diagnostics = {qid: {"counts": dict(c), "missing": sorted(expected[qid] - c.keys()),
                             "rare": [k for k, n in c.items() if n / c.total() < 0.05]} for qid, c in counts.items()}
        summary["partitions"][name] = {"records": len(rows), "sha256": digest(target), "overlong_ids": overlong, "labels": diagnostics}
        print(f"{name}: {len(rows)} records, {len(overlong)} over context; labels {json.dumps(diagnostics)}", flush=True)
    write_json(out / "summary.json", summary)
    # Fail before training rather than silently changing the evaluation population.
    if any(s["overlong_ids"] for s in summary["partitions"].values()):
        raise ValueError("records exceed the configured training context; inspect data/summary.json and increase train.max_state or shorten them")
    return summary


def training_arguments(config, checkpoint):
    from kev.train import parse_args
    full = config["mode"] == "full"
    meta, settings = checkpoint.meta, dict(config["train"])
    if not full and checkpoint.full: raise ValueError("LoRA mode needs a LoRA starting checkpoint")
    inherited = meta.extra.get("args", {})
    for key in ("lr", "batch", "accum", "checkpointing"):
        if settings[key] is None:
            if key not in inherited: raise ValueError(f"checkpoint has no {key}; set train.{key}")
            settings[key] = min(inherited[key], 5e-5) if key == "lr" else inherited[key]
    values = {"base": meta.base, "base_revision": meta.base_revision or "", "head_dim": meta.head_dim,
              "lora": meta.lora or 16, "lora_targets": inherited.get("lora_targets", "all"),
              "option_isolation": meta.option_isolation, "special_embeddings": False if full else meta.special_embeddings,
              "full_ft": full, "weights_dtype": "bf16" if full else meta.weights_dtype, "dtype": "bf16", "device": "cuda",
              "init_from": config["resolved_init"], "data": str(Path(config["output"]) / "data/train.jsonl"),
              "out": str(Path(config["output"]) / "checkpoint"),
              "p_none": 0, "p_none_distract": 0, "p_distract": 0, "p_none_pair": 0, "replay": 0,
              **{k: v for k, v in settings.items() if v is not None}}
    argv = [item for key, value in values.items() for item in (f"--{key}", str(int(value)) if isinstance(value, bool) else str(value))]
    return parse_args(argv)


def write_errors(rows, records, temperature, path, limit):
    from kev.metrics import probabilities_at_temperature
    from kev.suite import write_jsonl
    by_id = {r["_meta"]["id"]: r for r in records}
    errors = []
    for row in rows:
        p = probabilities_at_temperature(row, temperature)
        predicted = int(p.argmax())
        if predicted == row["label"]: continue
        record = by_id[row["id"]]
        errors.append({"id": row["id"], "question": row["question"], "confidence": float(p[predicted]),
                       "predicted": row["keys"][predicted], "label": row["keys"][row["label"]],
                       "probabilities": dict(zip(row["keys"], p.tolist())), "state": record["state"],
                       "instructions": record["questions"][row["question"]]["instructions"]})
    errors.sort(key=lambda r: -r["confidence"])
    write_jsonl(path, errors[:limit])


def evaluation_summary(report, rows, temperature):
    from kev.metrics import grouped_metrics
    return {"temperature": temperature, "raw": report["clean"], "calibrated": report["calibrated_clean"],
            "per_question": grouped_metrics(rows, "question", temperature), "n_questions": len(rows)}


def score_records(records, predictor, path, temperature, config, errors=False):
    from kev.benchmark import evaluate_records
    from kev.suite import write_json
    report, rows = evaluate_records(records, predictor, path, temperature)
    report["environment"] = predictor.environment
    write_json(Path(path) / "report.json", report)
    if errors or config["evaluation"]["errors_every_eval"]:
        write_errors(rows, records, temperature, Path(path) / "errors.jsonl", config["evaluation"]["error_limit"])
    return report, rows


def metric_event(out, values):
    with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(values, allow_nan=False) + "\n")
    print(EVENT + json.dumps(values, allow_nan=False), flush=True)


def worker(config):
    import numpy as np
    import torch
    from kev import train
    from kev.data import load_records
    from kev.model import training_context
    from kev.predictors import LocalPredictor
    out = Path(config["output"])
    development = load_records(out / "data/development.jsonl")
    context = training_context(config["training_args"]["max_state"])
    device = config["training_args"]["device"]

    def scheduler(optimizer, max_lrs, steps):
        # OneCycleLR's 10% warmup has a zero-length phase at exactly ten steps.
        warmup = 0.2 if steps == 10 else 0.1
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            config["scheduler"] = {"name": "OneCycleLR", "pct_start": warmup, "total_steps": max(steps, 1), "max_lr": max_lrs}
            yaml_write(out / "config.yaml", config)
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lrs, total_steps=max(steps, 1), pct_start=warmup)

    def on_step(model, tokenizer, progress):
        rank, step = progress["rank"], progress["step"]
        values = {"optimizer_step": step, **{f"train/{k}": progress[k] for k in ("loss", "lr", "grad_norm", "epoch")}}
        if step % config["evaluation"]["every_steps"] == 0:
            # All ranks execute identical forwards, so FSDP collectives match even for records
            # with different numbers of questions. Only rank 0 writes prediction files.
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            try:
                with LocalPredictor.from_model(model, tokenizer, device, context=context) as predictor:
                    if rank == 0:
                        report, _ = score_records(development, predictor, out / "periodic" / f"step-{step:07d}", 1.0, config)
                        values.update({f"validation/{k}": v for k, v in report["clean"].items() if isinstance(v, (int, float))})
                    else:
                        for record in development: predictor(record)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
        if rank == 0: metric_event(out, values)

    try:
        train.main(argparse.Namespace(**config["training_args"]), on_step=on_step,
                   distributed_lora=config["mode"] == "lora", scheduler_factory=scheduler)
    finally:
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


def score_on_device(run, config, directory, device, final=False):
    from kev.checkpoint import LoadOptions
    from kev.data import load_records
    from kev.device import empty_cache
    from kev.metrics import fit_temperature
    from kev.model import training_context
    from kev.predictors import LocalPredictor
    out = Path(config["output"])
    calibration = load_records(out / "data/calibration.jsonl")
    development = load_records(out / "data/development.jsonl")
    predictor = None
    try:
        predictor = LocalPredictor(run, device, LoadOptions(temperature=1.0, backend="torch"), context=training_context(config["training_args"]["max_state"]))
        _, calibration_rows = score_records(calibration, predictor, directory / "calibration", 1.0, config)
        temperature = fit_temperature(calibration_rows, aggregation="micro")
        report, rows = score_records(development, predictor, directory / "development", temperature, config, errors=final)
        if final: write_errors(rows, development, temperature, out / "errors.jsonl", config["evaluation"]["error_limit"])
        return evaluation_summary(report, rows, temperature), rows
    finally:
        del predictor
        gc.collect()
        empty_cache(device)


def score_checkpoint(run, config, directory, final=False):
    import torch
    from kev.device import empty_cache
    device = config["training_args"]["device"]
    try:
        return score_on_device(run, config, directory, device, final)
    except torch.cuda.OutOfMemoryError:
        if device != "cuda": raise
        # Keep partial GPU reports for diagnosis; the complete CPU read gets the normal path.
        if directory.exists(): directory.rename(directory.with_name(directory.name + "-cuda-oom"))
        print(f"{run}: evaluation exceeded GPU memory; retrying on CPU", flush=True)
    gc.collect()
    empty_cache("cuda")
    return score_on_device(run, config, directory, "cpu", final)


def launch_command(config, world):
    worker_args = [str(Path(__file__).resolve()), "--worker", str(Path(config["output"]) / "config.yaml")]
    if world == 1: return [sys.executable, *worker_args]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc_per_node={world}", *worker_args]


def print_comparison(result):
    base, final = result["baseline"], result["development"]
    columns = [base["raw"], base["calibrated"], final["raw"], final["calibrated"]]
    print(f"{'metric':30} {'baseline raw':>14} {'baseline cal':>14} {'final raw':>14} {'final cal':>14}")
    for key in ("acc", "brier", "nll", "ece", "confident_error_rate", "coverage_at_5pct_error", "mean_conf"):
        print(f"{key:30}" + "".join(f"{m[key]:14.4f}" for m in columns))
    for key, value in result["bootstrap"].items():
        print(f"{key} change: {value[f'micro_{key}_delta']:+.4f}; 95% CI {value['ci95']}")


def start_tracking(config):
    if not config["wandb"]["enabled"]: return None
    import wandb
    tracking = wandb.init(**{k: v for k, v in config["wandb"].items() if k != "enabled"}, dir=config["output"],
                          config={k: v for k, v in config.items() if k != "prepared_data"})
    tracking.define_metric("optimizer_step")
    tracking.define_metric("*", step_metric="optimizer_step")
    return tracking


def run(config):
    import torch
    from kev.checkpoint import Checkpoint, read_meta, write_meta
    from kev.experiment import git_commit, source_hashes
    from kev.metrics import paired_bootstrap, served_at
    from kev.model import load_tokenizer
    from kev.suite import digest, read_json, write_json
    from scripts.merge_lora_checkpoint import merge

    world = torch.cuda.device_count()  # CUDA handles indices, UUIDs, ordering, and an unset visibility mask.
    if not world: raise ValueError("no CUDA-visible GPUs; check CUDA_VISIBLE_DEVICES and the NVIDIA driver")
    out = Path(config["output"]).resolve()
    config["output"] = str(out)
    if out.exists(): raise FileExistsError(f"output already exists: {out}")
    initial = Checkpoint(config["init_from"])
    config["resolved_init"] = str(Path(initial.path).resolve())
    # Validate all training arguments before an expensive merge.
    args = training_arguments(config, initial)
    if args.row_budget and world > 1: raise ValueError("train.row_budget supports only one GPU; use length_sort and pass_tokens_max with multiple GPUs")
    config["training_args"] = vars(args)
    config["runtime"] = {"world_size": world, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "gpus": [torch.cuda.get_device_name(i) for i in range(world)], "torch": str(torch.__version__),
                         "python": sys.version, "effective_batch_size": args.batch * args.accum * world}
    config["provenance"] = {"kev_commit": git_commit(), "initial_checkpoint": config["resolved_init"], "base": initial.meta.base,
                            "base_revision": initial.meta.base_revision, "weights_sha256": initial.weights_sha256(),
                            "head_sha256": digest(initial.file("head.pt")),
                            "source_hashes": {**source_hashes(), **{str(p.relative_to(ROOT)): digest(p) for p in
                                              (Path(__file__).resolve(), ROOT / "scripts/merge_lora_checkpoint.py", ROOT / "scripts/interpolate_checkpoint.py")}}}
    out.mkdir(parents=True)
    yaml_write(out / "config.yaml", config)
    tokenizer = load_tokenizer(initial.meta.base, revision=initial.meta.base_revision)
    config["prepared_data"] = prepare_data(config, tokenizer)
    if config["prepared_data"]["partitions"]["train"]["records"] < world:
        raise ValueError("the training partition must contain at least one record per visible GPU")
    if config["mode"] == "full" and not initial.full:
        merged = merge(config["resolved_init"], out / "merged-init", weights_dtype="bf16")
        config["merge"] = merged
        config["resolved_init"] = merged["checkpoint"]
        config["training_args"] = vars(training_arguments(config, Checkpoint(merged["checkpoint"])))
    config["launch_command"] = launch_command(config, world)
    yaml_write(out / "config.yaml", config)
    tracking = None
    started = time.time()
    try:
        tracking = start_tracking(config)
        with (out / "train.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(config["launch_command"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
                try:
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                        if tracking is not None and line.startswith(EVENT): tracking.log(json.loads(line[len(EVENT):]))
                except BaseException:
                    proc.terminate()
                    raise
            if proc.returncode: raise subprocess.CalledProcessError(proc.returncode, config["launch_command"])
        final, rows = score_checkpoint(str(out / "checkpoint"), config, out / "final", final=True)
        baseline, baseline_rows = score_checkpoint(config["provenance"]["initial_checkpoint"], config, out / "baseline")
        temperature = final["temperature"]
        meta = read_meta(out / "checkpoint")
        meta.temperature = temperature
        meta.extra["temperature_fit"] = {"rows": "data/calibration.jsonl", "method": "min NLL, micro", "value": temperature}
        write_meta(out / "checkpoint", meta)
        result = {"development": final, "baseline": baseline, "wall_seconds": time.time() - started,
                  "bootstrap": {key: paired_bootstrap(served_at(rows, temperature), served_at(baseline_rows, baseline["temperature"]),
                                                      metric=key, aggregation="micro", samples=config["evaluation"]["bootstrap_samples"],
                                                      seed=config["evaluation"]["bootstrap_seed"]) for key in ("acc", "brier", "ece")}}
        write_json(out / "result.json", result)
        if tracking is not None:
            metrics = {"optimizer_step": read_json(out / "checkpoint/training_metrics.json")["optimizer_steps"]}
            for name, block in (("final", final), ("baseline", baseline)):
                for kind in ("raw", "calibrated"):
                    metrics.update({f"{name}/{kind}/{k}": v for k, v in block[kind].items() if isinstance(v, (int, float))})
            tracking.log(metrics)
        print_comparison(result)
    except BaseException:
        if tracking is not None: tracking.finish(exit_code=1)
        raise
    else:
        if tracking is not None: tracking.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--config", help="YAML configuration")
    options.add_argument("--example-config", action="store_true", help="print a complete example configuration")
    options.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.example_config:
        import yaml
        example = copy.deepcopy(DEFAULTS)
        example["output"], example["data"]["path"] = "runs/local-finetune", "data/labelled.jsonl"
        example["train"].update(lr=1e-5, batch=1, accum=4, checkpointing=True)
        print(yaml.safe_dump(example, sort_keys=False), end="")
    elif args.worker:
        worker(yaml_read(args.worker))
    else:
        config = configuration(yaml_read(args.config))
        config["output"] = str(next_output_dir(config["output"], Path(args.config).stem))
        run(config)


if __name__ == "__main__":
    main()
