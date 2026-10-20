"""Local fine-tuning integration on a tiny real Qwen3.5 backbone; no downloads or GPU required."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import finetune_local as local
from test_unit import tiny_base  # noqa: F401 - reuse the offline hybrid-backbone fixture


def settings(tmp_path, **changes):
    value = {"output": str(tmp_path / "run"), "data": {"path": "unused.jsonl"},
             "train": {"lr": 1e-4, "batch": 2, "accum": 1, "checkpointing": True}, **changes}
    return local.configuration(value)


def row(state="ticket", label="billing"):
    return {"state": state, "questions": {"team": {"type": "choice", "instructions": "which team",
            "criteria": {"billing": None, "shipping": None}, "label": label}}}


def test_configuration_requires_full_settings_and_rejects_typos(tmp_path):
    for key in ("lr", "batch", "accum", "checkpointing"):
        cfg = settings(tmp_path)
        cfg["train"][key] = None
        with pytest.raises(ValueError, match=f"train.{key}"): local.configuration(cfg)
    with pytest.raises(ValueError, match="unknown train"): settings(tmp_path, train={"learning_rate": 0.1})
    cfg = settings(tmp_path, mode="lora", train={})
    assert cfg["evaluation"]["every_steps"] == 10 and not cfg["wandb"]["enabled"]
    cfg["evaluation"]["errors_every_eval"] = "false"
    with pytest.raises(ValueError, match="boolean"): local.configuration(cfg)


def test_training_defaults_and_yaml_round_trip(tmp_path):
    from kev.checkpoint import Meta
    meta = Meta(base="tiny", lora=4, extra={"args": {"lr": 2e-4, "batch": 3, "accum": 2, "checkpointing": 1, "lora_targets": "qv"}})
    checkpoint = SimpleNamespace(meta=meta, full=False)
    cfg = settings(tmp_path, mode="lora", train={})
    cfg["resolved_init"] = "initial"
    args = local.training_arguments(cfg, checkpoint)
    assert (args.lr, args.batch, args.accum, args.checkpointing) == (5e-5, 3, 2, 1)
    assert args.lora_targets == "qv" and args.device == "cuda"
    assert (args.p_none, args.p_none_distract, args.p_distract, args.p_none_pair, args.replay) == (0, 0, 0, 0, 0)
    full = settings(tmp_path)
    full["resolved_init"] = "initial"
    full_args = local.training_arguments(full, checkpoint)
    assert full_args.full_ft and full_args.weights_dtype == "bf16" and full_args.lr == 1e-4
    cfg["training_args"] = vars(args)
    local.yaml_write(tmp_path / "config.yaml", cfg)
    assert local.yaml_read(tmp_path / "config.yaml") == cfg
    command = local.launch_command(cfg, 3)
    assert "--nproc_per_node=3" in command and "torch.distributed.run" in command
    assert "torch.distributed.run" not in local.launch_command(cfg, 1)


def test_conversion_nested_fields_multiple_inputs_and_bad_labels(tmp_path):
    from kev.suite import write_json, write_jsonl
    spec = {"questions": {**row()["questions"], "urgent": {"type": "noul", "instructions": "urgent?"},
                          "rating": {"type": "score", "instructions": "severity?", "criteria": ["low", "high"]}}}
    write_json(tmp_path / "spec.json", spec)
    records = [{"subject": "refund", "body": "charged twice", "meta": {"team": "Billing Ops"}, "urgent": "yes", "rating": 2},
               {"subject": "missing", "body": "lost", "meta": {"team": "unknown"}, "urgent": "no", "rating": 1}]
    write_jsonl(tmp_path / "source.jsonl", records)
    data = {**local.DEFAULTS["data"], "spec": str(tmp_path / "spec.json"), "state": ["subject", "body"],
            "labels": {"team": "meta.team"}, "mappings": {"team": {"Billing Ops": "billing"}}, "score_offset": 1}
    kept, rejected = local.read_data(tmp_path / "source.jsonl", data, convert=True)
    assert len(kept) == 1 and sum(rejected.values()) == 1
    assert kept[0]["state"] == {"subject": "refund", "body": "charged twice"}
    assert [q["label"] for q in kept[0]["questions"].values()] == ["billing", True, 1]
    with pytest.raises(ValueError, match="integer"):
        local.converted_label(spec["questions"]["rating"], "1.5", {}, 0)


def test_invalid_jsonl_is_counted_without_losing_valid_rows(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(row()) + "\n{bad\n" + json.dumps(row("second")) + "\n", encoding="utf-8")
    records, rejected = local.read_data(path, local.DEFAULTS["data"])
    assert len(records) == 2 and sum(rejected.values()) == 1
    assert next(iter(rejected)).startswith("invalid JSON")


def test_cleanup_and_state_grouped_splitting():
    records = [row("same"), row("same"), row(" SAME ", "shipping"), *[row(str(i)) for i in range(30)]]
    kept, report = local.clean_records(records)
    assert len(kept) == 30 and report == {"duplicates_dropped": 1, "conflicting_states_dropped": 1}
    kept.extend([row("0", "shipping")])
    parts = local.split_records(kept, 0.15, 0.15, 3)
    assert parts == local.split_records(kept, 0.15, 0.15, 3)
    sets = [set(local.state_key(r["state"]) for r in rows) for rows in parts.values()]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])


def test_data_preparation_holdout_and_overlap(tiny_base, tmp_path):
    from kev.model import load_tokenizer, MAX_STATE
    from kev.suite import write_jsonl
    cfg = settings(tmp_path)
    Path(cfg["output"]).mkdir()
    train = [row(f"train {i}") for i in range(30)]
    held = [row(f"real {i}") for i in range(12)]
    write_jsonl(tmp_path / "train.jsonl", train + held[:2])
    write_jsonl(tmp_path / "holdout.jsonl", held)
    cfg["data"].update(path=str(tmp_path / "train.jsonl"), holdout=str(tmp_path / "holdout.jsonl"))
    cfg["training_args"] = {"max_state": MAX_STATE}
    tok = load_tokenizer(tiny_base / "base")
    summary = local.prepare_data(cfg, tok)
    assert summary["holdout_overlap_dropped"] == 2
    assert [summary["partitions"][p]["records"] for p in local.PARTITIONS] == [30, 6, 6]
    cfg["output"] = str(tmp_path / "other")
    Path(cfg["output"]).mkdir()
    cfg["data"].update(path=str(tmp_path / "prepared"), holdout=None)
    (tmp_path / "prepared").mkdir()
    for part in local.PARTITIONS: write_jsonl(tmp_path / "prepared" / f"{part}.jsonl", train)
    with pytest.raises(ValueError, match="state overlap"): local.prepare_data(cfg, tok)


def test_borrowed_predictor_restores_rng_modes_and_temperature_on_error(tiny_base):
    from kev.model import DecisionModel, load_tokenizer
    from kev.predictors import LocalPredictor
    tok = load_tokenizer(tiny_base / "base")
    model = DecisionModel(str(tiny_base / "base"), tok, "cpu", lora=4)
    model.train()
    model.head.eval()
    model.head.temperature = 2.5
    modes = [m.training for m in model.modules()]
    state = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="deliberate"):
        with LocalPredictor.from_model(model, tok, "cpu") as predictor:
            assert predictor.model is model and not model.training and model.head.temperature == 1
            torch.rand(9)
            raise RuntimeError("deliberate")
    assert modes == [m.training for m in model.modules()]
    assert model.head.temperature == 2.5 and torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("full", [False, True])
def test_periodic_evaluation_does_not_change_training_weights(tiny_base, tmp_path, full):
    from safetensors.torch import load_file
    from kev import train
    from kev.checkpoint import read_meta
    from kev.data import load_records
    from kev.predictors import LocalPredictor
    common = ["--base", str(tiny_base / "base"), "--data", str(tiny_base / "data.jsonl"), "--device", "cpu",
              "--batch", "2", "--accum", "1", "--lr", "1e-3", "--max_steps", "3", "--checkpointing", "1"]
    if full: common += ["--full_ft", "1", "--weights_dtype", "bf16"]
    records = load_records(tiny_base / "data.jsonl")[:2]
    events = []

    def evaluate(model, tok, info):
        with LocalPredictor.from_model(model, tok, "cpu") as predictor:
            for record in records:
                assert predictor(record)["inference_temperature"] == 1
        events.append(info)

    for name, callback in (("plain", None), ("evaluated", evaluate)):
        train.main(train.parse_args([*common, "--out", str(tmp_path / name)]), on_step=callback)
    filename = "model.safetensors" if full else "adapter_model.safetensors"
    a, b = (load_file(tmp_path / name / filename) for name in ("plain", "evaluated"))
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    a, b = (read_meta(tmp_path / name).head for name in ("plain", "evaluated"))
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert [e["step"] for e in events] == [1, 2, 3] and all(e["loss"] > 0 for e in events)


@pytest.mark.parametrize("full", [False, True])
def test_two_ranks_evaluate_and_continue_with_identical_weights(tiny_base, tmp_path, full):
    from safetensors.torch import load_file
    from kev import train
    from kev.checkpoint import read_meta
    from kev.suite import read_json
    common = ["--base", str(tiny_base / "base"), "--data", str(tiny_base / "data.jsonl"), "--device", "cpu",
              "--batch", "2", "--accum", "1", "--lr", "1e-3", "--max_steps", "3", "--checkpointing", "1"]
    if full: common += ["--full_ft", "1", "--weights_dtype", "bf16"]
    for name, interval in (("plain", 10), ("evaluated", 1)):
        out = tmp_path / name
        (out / "data").mkdir(parents=True)
        (out / "data/development.jsonl").write_bytes((tiny_base / "data.jsonl").read_bytes())
        cfg = settings(tmp_path, mode="full" if full else "lora")
        cfg["output"] = str(out)
        cfg["evaluation"]["every_steps"] = interval
        cfg["training_args"] = vars(train.parse_args([*common, "--out", str(out / "checkpoint")]))
        local.yaml_write(out / "config.yaml", cfg)
        result = subprocess.run(local.launch_command(cfg, 2), env={**os.environ, "OMP_NUM_THREADS": "1"},
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    filename = "model.safetensors" if full else "adapter_model.safetensors"
    a, b = (load_file(tmp_path / name / "checkpoint" / filename) for name in ("plain", "evaluated"))
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    a, b = (read_meta(tmp_path / name / "checkpoint").head for name in ("plain", "evaluated"))
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert len(list((tmp_path / "evaluated/periodic").glob("step-*/report.json"))) == 3
    assert not list((tmp_path / "evaluated").rglob("errors.jsonl"))
    assert read_json(tmp_path / "evaluated/checkpoint/training_metrics.json")["world_size"] == 2


def test_final_calibration_error_policy_and_bootstrap(tiny_base, tmp_path):
    from kev import train
    from kev.checkpoint import Checkpoint
    from kev.metrics import paired_bootstrap, served_at
    from kev.suite import read_json
    from scripts.merge_lora_checkpoint import merge
    checkpoint = tmp_path / "initial"
    train.main(train.parse_args(["--base", str(tiny_base / "base"), "--data", str(tiny_base / "data.jsonl"),
                                "--device", "cpu", "--max_steps", "1", "--out", str(checkpoint)]))
    cfg = settings(tmp_path)
    out = Path(cfg["output"])
    (out / "data").mkdir(parents=True)
    # Separate states, with intentionally different labels, exercise independent calibration/dev reads.
    records = (tiny_base / "data.jsonl").read_text(encoding="utf-8").splitlines()
    for part, selected in (("calibration", records[:4]), ("development", records[4:8])):
        (out / "data" / f"{part}.jsonl").write_text("\n".join(selected) + "\n", encoding="utf-8")
    cfg["resolved_init"] = str(checkpoint)
    cfg["training_args"] = vars(local.training_arguments(cfg, Checkpoint(checkpoint)))
    cfg["training_args"]["device"] = "cpu"
    merged = merge(str(checkpoint), tmp_path / "merged", weights_dtype="bf16", log=lambda _: None)
    assert Checkpoint(merged["checkpoint"]).full
    baseline, baseline_rows = local.score_checkpoint(str(checkpoint), cfg, out / "baseline")
    final, rows = local.score_checkpoint(merged["checkpoint"], cfg, out / "final", final=True)
    assert len(rows) == len(baseline_rows) == 8
    assert (out / "errors.jsonl").exists() and not list((out / "baseline").rglob("errors.jsonl"))
    assert [r["id"] for r in rows] == [r["id"] for r in baseline_rows]
    for metric in ("acc", "brier", "ece"):
        result = paired_bootstrap(served_at(rows, final["temperature"]), served_at(baseline_rows, baseline["temperature"]),
                                  metric=metric, aggregation="micro", samples=20)
        assert len(result["ci95"]) == 2
    cfg["evaluation"]["errors_every_eval"] = True
    local.score_checkpoint(str(checkpoint), cfg, out / "every")
    assert len(list((out / "every").rglob("errors.jsonl"))) == 2
    assert read_json(out / "final/development/report.json")["environment"]["device"] == "cpu"


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_complete_launcher_with_opt_in_tracking(tiny_base, tmp_path, monkeypatch, mode):
    """Use real CPU training, checkpoint merging, scoring, calibration, and subprocess logging.
    Only CUDA discovery and the external tracking service are replaced in this integration test.
    """
    from kev import train
    from kev.checkpoint import read_meta
    from kev.suite import read_json
    initial = tmp_path / "initial"
    train.main(train.parse_args(["--base", str(tiny_base / "base"), "--data", str(tiny_base / "data.jsonl"),
                                "--device", "cpu", "--max_steps", "1", "--out", str(initial)]))
    original = local.training_arguments

    def cpu_args(config, checkpoint):
        args = original(config, checkpoint)
        args.device, args.dtype = "cpu", "fp32"
        return args

    monkeypatch.setattr(local, "training_arguments", cpu_args)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "CPU test fixture")
    events, starts, finishes = [], [], []
    tracker = SimpleNamespace(define_metric=lambda *a, **k: None, log=events.append, finish=lambda **k: finishes.append(k))

    def start(**kwargs):
        starts.append(kwargs)
        return tracker

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=start))
    cfg = settings(tmp_path, mode=mode, init_from=str(initial), data={"path": str(tiny_base / "data.jsonl")},
                   evaluation={"every_steps": 2, "bootstrap_samples": 20}, wandb={"enabled": mode == "full", "mode": "offline"})
    local.run(cfg)
    out = Path(cfg["output"])
    saved = local.yaml_read(out / "config.yaml")
    result = read_json(out / "result.json")
    assert saved["training_args"]["device"] == "cpu" and saved["runtime"]["effective_batch_size"] == 2
    assert "scripts/finetune_local.py" in saved["provenance"]["source_hashes"]
    assert read_meta(out / "checkpoint").temperature == result["development"]["temperature"]
    assert (out / "merged-init/checkpoint").exists() == (mode == "full")
    assert (out / "train.log").stat().st_size and (out / "metrics.jsonl").stat().st_size
    assert not list((out / "periodic").rglob("errors.jsonl")) and (out / "errors.jsonl").exists()
    if mode == "full":
        assert len(starts) == 1 and len(finishes) == 1 and events
        assert any("train/loss" in e for e in events) and any("validation/acc" in e for e in events)
        assert "final/calibrated/acc" in events[-1]
    else:
        assert not starts and not events and not finishes


def test_checkpoint_scoring_falls_back_only_on_cuda_oom(tmp_path, monkeypatch):
    cfg = settings(tmp_path)
    cfg["training_args"] = {"device": "cuda"}
    directory = tmp_path / "evaluation"
    devices = []

    def score(run, config, path, device, final):
        devices.append(device)
        if device == "cuda":
            path.mkdir()
            (path / "partial.txt").write_text("incomplete", encoding="utf-8")
            raise torch.cuda.OutOfMemoryError("test")
        return "complete"

    monkeypatch.setattr(local, "score_on_device", score)
    assert local.score_checkpoint("unused", cfg, directory) == "complete"
    assert devices == ["cuda", "cpu"]
    assert (tmp_path / "evaluation-cuda-oom/partial.txt").exists()

    def bad(*args): raise ValueError("bad data")
    monkeypatch.setattr(local, "score_on_device", bad)
    with pytest.raises(ValueError, match="bad data"): local.score_checkpoint("unused", cfg, directory)


def test_wandb_offline_logging(tmp_path):
    pytest.importorskip("wandb")
    cfg = settings(tmp_path, wandb={"enabled": True, "mode": "offline"})
    Path(cfg["output"]).mkdir()
    tracking = local.start_tracking(cfg)
    try:
        tracking.log({"optimizer_step": 1, "train/loss": 0.5, "validation/acc": 0.75})
        assert tracking.config["evaluation"]["every_steps"] == 10
        directory = Path(tracking.dir)
    finally:
        tracking.finish()
    assert list(directory.parent.glob("*.wandb"))


@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("gpus", [1, 2])
def test_cuda_launcher(tiny_base, tmp_path, mode, gpus):
    if not torch.cuda.is_available() or torch.cuda.device_count() < gpus:
        pytest.skip(f"needs {gpus} CUDA GPUs")
    from kev import train
    from kev.checkpoint import read_meta
    from kev.suite import read_json
    initial = tmp_path / "initial"
    train.main(train.parse_args(["--base", str(tiny_base / "base"), "--data", str(tiny_base / "data.jsonl"),
                                "--device", "cpu", "--max_steps", "1", "--out", str(initial)]))
    cfg = settings(tmp_path, mode=mode, init_from=str(initial), data={"path": str(tiny_base / "data.jsonl")},
                   train={"lr": 1e-4, "batch": 2, "accum": 1, "checkpointing": True, "max_steps": 3},
                   evaluation={"every_steps": 1, "bootstrap_samples": 20, "errors_every_eval": True})
    local.yaml_write(tmp_path / "input.yaml", cfg)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = visible.split(",") if visible is not None else [str(i) for i in range(torch.cuda.device_count())]
    command = [sys.executable, str(Path(local.__file__)), "--config", str(tmp_path / "input.yaml")]
    result = subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(visible[:gpus]), "OMP_NUM_THREADS": "1"},
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    out = Path(cfg["output"])
    metrics = read_json(out / "checkpoint/training_metrics.json")
    assert metrics["world_size"] == gpus and metrics["optimizer_steps"] == 3
    assert len(list((out / "periodic").glob("step-*/errors.jsonl"))) == 3
    report = read_json(out / "final/development/report.json")
    assert report["environment"]["device"] == "cuda"
    assert read_meta(out / "checkpoint").temperature == read_json(out / "result.json")["development"]["temperature"]


@pytest.mark.parametrize("full", [False, True])
def test_ten_step_run_reaches_default_validation(tiny_base, tmp_path, full):
    from kev import train
    from kev.suite import read_json
    cfg = settings(tmp_path, mode="full" if full else "lora")
    out = Path(cfg["output"])
    (out / "data").mkdir(parents=True)
    (out / "data/development.jsonl").write_bytes((tiny_base / "data.jsonl").read_bytes())
    args = ["--base", str(tiny_base / "base"), "--data", str(tiny_base / "data.jsonl"), "--device", "cpu",
            "--batch", "1", "--accum", "1", "--lr", "1e-3", "--max_steps", "10", "--out", str(out / "checkpoint")]
    if full: args += ["--full_ft", "1", "--weights_dtype", "bf16"]
    cfg["training_args"] = vars(train.parse_args(args))
    local.worker(cfg)
    assert read_json(out / "checkpoint/training_metrics.json")["optimizer_steps"] == 10
    assert len(list((out / "periodic").glob("step-*/report.json"))) == 1
    assert (out / "periodic/step-0000010/report.json").exists()
    assert local.yaml_read(out / "config.yaml")["scheduler"]["pct_start"] == 0.2
