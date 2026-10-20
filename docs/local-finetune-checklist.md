# Local fine-tuning implementation checklist

Status: implementation complete. Verification results and limitations are below.

Entry point: `scripts/finetune_local.py`. Optional hooks are in `kev/train.py` and
`kev/predictors.py`; existing callers keep their defaults.

Deliver one Python entry point for local CUDA training, based on the pipeline in
`skills/kev-finetune/scripts/kev_modal.py`. The kev-finetune skill is reference
material only, not an invoked workflow. The decisions below were confirmed before implementation.

## Training and configuration

- [x] Load a local or Hub Kev checkpoint and preserve its pointer head and architecture metadata.
- [x] Default to full fine-tuning through `kev.full_ft`; provide a LoRA configuration option.
- [x] Automatically merge a starting LoRA checkpoint into bf16 full weights before full fine-tuning.
- [x] In LoRA mode, inherit training defaults with explicit overrides. Full mode requires separate lr, batch, accum, and checkpointing settings. Original feature 3.
- [x] Train across all CUDA-visible GPUs in both full and LoRA modes; support one visible GPU.
- [x] Interpret batch size per GPU and record effective global batch size.
- [x] Save all resolved configuration, including inherited defaults, data processing, training, evaluation, and logging options, in YAML in the output directory. Exclude authentication secrets.
- [x] Support Weights & Biases logging for configuration, training loss, learning rate, and evaluation metrics.
- [x] Save checkpoint provenance, data provenance, GPU and software information, elapsed time, and training logs. Original feature 17.
- [x] Save a loadable final checkpoint and its tokenizer.
- [x] Handle the inherited ten-step OneCycleLR division-by-zero case inside the
  script through an optional scheduler factory; log the actual schedule in YAML.

## Data preparation

- [x] Accept native Kev JSONL records and prepared data partitions.
- [x] Convert CSV, TSV, JSON, and JSONL using question specifications, input columns, label mappings, nested fields, and score offsets. Original feature 14.
- [x] Validate records and labels; handle duplicate records and conflicting-label states; report label distribution and input-length issues. Original feature 13.
- [x] Split deterministically by normalized input state into train, calibration, and development partitions; prevent state overlap. Original feature 12.
- [x] Support a separate real-data holdout for calibration and development. Original feature 12.
- [x] Save prepared partitions and data-processing summaries.
- [x] Disable none-answer, distractor, and none-pair augmentation. The existing choice-order shuffling remains.

## Evaluation and reporting

- [x] Evaluate periodically during training through a `LocalPredictor` object, at a configurable interval.
- [x] Preserve optimizer, scheduler, RNG, training mode, and kernel settings across periodic evaluation.
- [x] Borrow the same model for periodic evaluation on all ranks, without allocating a second model.
- [x] Fit temperature on calibration data only and store the final fitted temperature in the final checkpoint. Original feature 4.
- [x] Run final development-set evaluation and save predictions and metrics. Original feature 5.
- [x] Evaluate the starting checkpoint on the same data and fit its own calibration temperature. Original feature 6.
- [x] Compute paired bootstrap confidence intervals for accuracy, Brier score, and ECE differences. Original feature 7.
- [x] Report raw and calibrated metrics, per-question metrics, confident errors, and selective coverage. Original feature 9.
- [x] Generate confidence-sorted error inspection output for the final development evaluation by default. Original feature 10.
- [x] Provide an option to generate error inspection output on every evaluation.

## Verification

- [x] Test conversion, validation, splitting, overlap handling, configuration resolution, and YAML output.
- [x] Test calibration separation, baseline comparisons, bootstrap reports, and error-report scheduling.
- [x] Test distributed launch selection and logging behavior.
- [x] Verify training and periodic evaluation on local GPUs for full fine-tuning and LoRA, including multiple visible GPUs where available.
- [x] Run applicable repository checks and record verification blocked by dependencies.
- [x] Include usage and configuration examples in the script.

## Excluded features

Do not add public replay, none-pair augmentation, public-suite regression checks,
standalone evaluation/comparison commands, synthetic data generation, workload
discovery, dataset sizing, Hub publishing, serving, or Modal infrastructure.
These correspond to unselected original features 1, 2, 8, 11, 15, 16, 18, 19,
and 20. Baseline comparisons within the training workflow remain included.

## Confirmed decisions

- Supporting Kev changes must be opt-in and preserve existing defaults.
- Automatically merge starting adapters for full fine-tuning.
- Full fine-tuning requires explicit lr, batch, accum, and checkpointing settings.
  Batch size is per GPU. LoRA mode retains checkpoint-derived defaults.
- Evaluate every 10 optimizer steps by default, at temperature 1. Fit temperature
  only after training, on the calibration partition.
- Reuse the live model for periodic evaluation. This works with both replicated
  LoRA and FSDP full fine-tuning. All ranks score the same records to keep
  collectives aligned; only rank 0 writes reports. No extra model is loaded.
- Disable the trainer's none-answer and distractor augmentation probabilities.
- W&B stays disabled unless enabled in YAML.

## Verification record

- Tiny Qwen3.5, one and two CPU workers, both full and LoRA: periodic evaluation
  leaves all final backbone/adapter and head tensors bit-identical to runs
  without periodic evaluation.
- Tiny Qwen3.5, one and two H100 NVL GPUs, both full and LoRA: four complete
  launcher smoke tests passed, including merging where applicable, three
  optimizer steps, periodic errors, final calibration, and baseline comparison.
- W&B 0.31.0: offline logging tested with the actual SDK, using a temporary
  installation without changing project dependencies or uploading artifacts.
- Default trainer parity against `origin/main` at
  `dd375f27fd61486c52204205e10a0d2e10f3b34a`: tiny Qwen3.5, three CPU steps,
  checkpointing, none pairs, permutation KL, and ordinal loss. All 30 LoRA
  tensors and all 27 full-weight tensors match exactly; maximum backbone,
  adapter, and pointer-head difference is 0.0.
- The new tests cover data conversion and cleanup, splitting and overlap,
  YAML configuration, inherited defaults, calibration, bootstrap comparisons,
  error-report scheduling, CPU fallback, tracking enablement, and launcher
  integration. Seventeen CPU tests passed, plus the offline W&B test and four
  CUDA tests above.
- Exactly ten optimizer steps now complete in both modes and produce the default
  step-10 validation report. Other run lengths keep the trainer's 10% warmup;
  the ten-step case uses 20% to avoid PyTorch's zero-length warmup phase.
- The repository regression suite passed with 492 passed and 26 skipped.
  The separate `tests/test_model.py` run is blocked by unavailable model
  artifacts: five tests skipped for the missing smoke checkpoint, one passed,
  and the next failed because Qwen/Qwen2.5-0.5B weights are absent from the
  local cache. The bounded retry used `HF_HUB_OFFLINE=1`.

Final/baseline evaluation loads checkpoints after the training processes exit.
If a checkpoint cannot be scored on one GPU, it retries on CPU and preserves
partial GPU reports under a directory ending in `-cuda-oom`.

Run the focused tests with:

```bash
OMP_NUM_THREADS=1 uv run --with pyyaml --with wandb python -m pytest tests/test_finetune_local.py -q
```

CUDA cases skip if the required GPUs are unavailable; the remaining model tests
use the small generated Qwen3.5 fixture and need no model downloads.
