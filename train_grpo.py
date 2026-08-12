"""GRPO fine-tuning, continuing from an SFT checkpoint produced by train_sft.py.

    python train_grpo.py
    python train_grpo.py --model-key qwen3.5-4b --reasoning-format new

Reads the SFT run's merged checkpoint from
outputs/<model-key>-sft-<reasoning-format>/merged (override with
--sft-checkpoint if it lives elsewhere), and writes
outputs/<model-key>-grpo-<reasoning-format>/(merged).

The "new" reasoning format is required here: GRPO's reward functions score
the relevance_reasoning -> raw_data -> derivation_reasoning structure, so
training data without those fields is dropped by data_prep.py's
build_grpo_row().
"""

import argparse
import gc
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import GRPOConfig, GRPOTrainer

from common.paths import DATA_DIR, OUTPUTS_DIR
from data_prep import build_grpo_splits
from model_registry import (
    MODEL_REGISTRY,
    grpo_run_name,
    sft_merged_dir,
    spec_from_checkpoint,
)
from reward_funcs import DEFAULT_REWARD_WEIGHTS, REWARD_FUNCS, reward_weights_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--model-key", default="qwen3.5-4b", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--reasoning-format", default="new", choices=["old", "new"])
    parser.add_argument(
        "--sft-checkpoint", type=Path, default=None,
        help="Defaults to outputs/<model-key>-sft-<reasoning-format>/merged.",
    )

    parser.add_argument("--train-jsonl", type=Path, default=DATA_DIR / "train_grpo.jsonl")
    parser.add_argument("--val-jsonl", type=Path, default=DATA_DIR / "val.jsonl")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the auto-generated output dir.")

    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=5)

    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-train-batch", type=int, default=1)
    parser.add_argument("--per-device-eval-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.05)

    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    for reward_fn in REWARD_FUNCS:
        name = reward_fn.__name__
        parser.add_argument(
            f"--reward-weight-{name.replace('_', '-')}",
            type=float,
            default=DEFAULT_REWARD_WEIGHTS[name],
            dest=f"reward_weight_{name}",
        )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sft_checkpoint = args.sft_checkpoint or sft_merged_dir(args.outputs_dir, args.model_key, args.reasoning_format)
    if not sft_checkpoint.exists():
        raise FileNotFoundError(
            f"No SFT checkpoint at {sft_checkpoint}. Run train_sft.py first, or pass --sft-checkpoint."
        )
    spec = spec_from_checkpoint(args.model_key, sft_checkpoint)
    output_dir = args.output_dir or (args.outputs_dir / grpo_run_name(args.model_key, args.reasoning_format))

    datasets = build_grpo_splits(train_jsonl=args.train_jsonl, val_jsonl=args.val_jsonl, data_dir=args.data_dir)
    train_dataset, val_dataset = datasets["train"], datasets["val"]

    device_map = {"": 0} if torch.cuda.is_available() else None

    processor = AutoProcessor.from_pretrained(
        spec.model_id,
        trust_remote_code=spec.trust_remote_code,
        **spec.processor_kwargs,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        spec.model_id,
        device_map=device_map,
        dtype=torch.bfloat16,
        trust_remote_code=spec.trust_remote_code,
        attn_implementation=spec.attn_implementation,
    )

    lora_config = None
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=spec.lora_target_modules,
            task_type="CAUSAL_LM",
        )

    reward_weights = {name: getattr(args, f"reward_weight_{name}") for name in DEFAULT_REWARD_WEIGHTS}

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch,
        per_device_eval_batch_size=args.per_device_eval_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        reward_weights=reward_weights_list(reward_weights),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=REWARD_FUNCS,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        peft_config=lora_config,
    )

    try:
        trainer.train()
    except (KeyboardInterrupt, Exception):
        print("trainer.train() interrupted or failed -- releasing GPU memory.")
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        raise

    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    merged_dir = output_dir / "merged"
    if args.use_lora:
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        processor.save_pretrained(str(merged_dir))
        processor.image_processor.save_pretrained(str(merged_dir))

        del merged_model, trainer, model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        merged_dir = output_dir

    print(f"Saved merged model -> {merged_dir}")


if __name__ == "__main__":
    main()
