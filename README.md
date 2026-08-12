# ImplicitIR

Code for the **ImplicitI**nformation**R**etrieval paper. Fine-tunes a vision-language model (VLM) to answer document questions that require implicit reasoning rather than plain OCR transcription.

The benchmark itself lives in a separate repo: [ImplicitIR-Benchmark](https://github.com/VXRealLimited/ImplicitIR-Benchmark). This repo covers training; see below for the two stages.

Two stages:

1. **SFT** on curated reasoning traces (`train_sft.py`)
2. **GRPO** on top of the SFT checkpoint, with reward functions that check grounding in the source document and arithmetic correctness (`train_grpo.py`)

The merged checkpoints this produces are meant to be served and scored by **ImplicitIR-Benchmark** — see [Handing off to eval](#handing-off-to-eval)
below.

## Setup

```
pip install -r requirements.txt
```

Training pulls the base model from Hugging Face (`model_registry.py::MODEL_REGISTRY`). If it's gated, authenticate first:

```
hf auth login
```

`data/train_sft.jsonl`, `train_grpo.jsonl`, and `val.jsonl` are already-curated, non-overlapping splits committed to the repo — each line is one document with an `"image"` path (relative to `data/`) and a list of `"samples"` (question/answer pairs, optionally with `relevance_reasoning` / `raw_data` / `derivation_reasoning` for the reasoning format GRPO needs).

## Usage

### Prepare data only

Flattens the doc-level jsonl files into row-per-question HF datasets and caches them at `data/sft_splits/*.jsonl`. `train_sft.py` does this automatically as part of its own run; running it standalone is only useful to inspect the flattened form without touching a GPU:

```
python data_prep.py
```

### Supervised fine-tuning

```
python train_sft.py --model-key qwen3.5-4b --reasoning-format new
```

`--reasoning-format old` trains on `<think>{derivation}</think>\n{answer}`; `--reasoning-format new` trains on the richer `<think>{relevance}<raw>{raw_data}</raw>{derivation}</think>{"answer": ...}` structure that GRPO's reward functions later check for. Run `--help` for the full list of hyperparameters (LoRA rank, batch size, epochs, 4-bit, etc.). Produces `outputs/<model-key>-sft-<reasoning-format>/merged/`.

### GRPO

Continues from an SFT checkpoint (default: the matching `<model-key>-sft-<reasoning-format>` run's merged output). Requires `--reasoning-format new` data, since the reward functions in `reward_funcs.py` score the relevance/raw-data/derivation structure:

```
python train_grpo.py --model-key qwen3.5-4b --reasoning-format new
```

Reward weights are configurable per-function, e.g. `--reward-weight-answer-accuracy 3.0`. Produces `outputs/<model-key>-grpo-<reasoning-format>/merged/`.

## Adding a new model

Add an entry to `MODEL_REGISTRY` in `model_registry.py`: the HF model id, LoRA target modules, and any processor kwargs it needs.

## Handing off to eval

[ImplicitIR-Benchmark](https://github.com/VXRealLimited/ImplicitIR-Benchmark) serves and scores checkpoints produced here. The two repos are independent, so copy or symlink the relevant `outputs/<run-name>/merged/` directory to wherever `ImplicitIR-Benchmark`'s `docker-compose.yml` expects it, or edit that file's volume paths to point at this checkpoint directly.

## License

CC BY 4.0 — see [LICENSE](LICENSE).