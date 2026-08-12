"""Supervised fine-tuning of a VLM on the ImplicitIR document reasoning data.

    python train_sft.py 
    python train_sft.py --model-key qwen3.5-4b --reasoning-format old

Produces outputs/<model-key>-sft-<reasoning-format>/ (LoRA adapter + trainer
state) and .../merged/ (the base model with the adapter merged in, ready to
be served by vLLM or continued into GRPO).
"""

import argparse
import gc
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from common.paths import DATA_DIR, OUTPUTS_DIR, SPLITS_DIR
from data_prep import build_sft_splits
from model_registry import MODEL_REGISTRY, sft_run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--model-key", default="qwen3.5-4b", choices=sorted(MODEL_REGISTRY))
    parser.add_argument(
        "--reasoning-format", default="new", choices=["old", "new"],
        help='"old": <think>{thinking}</think>\\n{answer}. '
             '"new": <think>{relevance_reasoning}<raw>{raw_data}</raw>{derivation_reasoning}</think>{"answer": ...}',
    )

    parser.add_argument("--train-jsonl", type=Path, default=DATA_DIR / "train_sft.jsonl")
    parser.add_argument("--val-jsonl", type=Path, default=DATA_DIR / "val.jsonl")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the auto-generated output dir.")

    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--lr-scheduler-type", default="cosine")

    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    parser.add_argument("--use-4bit", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = MODEL_REGISTRY[args.model_key]
    output_dir = args.output_dir or (args.outputs_dir / sft_run_name(args.model_key, args.reasoning_format))

    datasets = build_sft_splits(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        reasoning_format=args.reasoning_format,
    )
    train_dataset, val_dataset = datasets["train"], datasets["val"]

    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

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
        quantization_config=quant_config,
        attn_implementation=spec.attn_implementation,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = None
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=spec.lora_target_modules,
            task_type="CAUSAL_LM",
        )

    if len(train_dataset):
        sample = train_dataset[0]
        print("Sample prompt:", sample["prompt"])
        print("Sample completion:", sample["completion"])
        print("Sample images:", sample["images"])

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        max_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        weight_decay=args.weight_decay,
        completion_only_loss=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
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
        if args.use_4bit:
            base_model = AutoModelForImageTextToText.from_pretrained(
                spec.model_id, dtype=torch.bfloat16, trust_remote_code=spec.trust_remote_code,
            )
            merged_model = PeftModel.from_pretrained(base_model, str(output_dir)).merge_and_unload()
        else:
            merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        processor.save_pretrained(str(merged_dir))
        processor.image_processor.save_pretrained(str(merged_dir))

        del merged_model, trainer, model
        if args.use_4bit:
            del base_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        merged_dir = output_dir

    print(f"Saved merged model -> {merged_dir}")


if __name__ == "__main__":
    main()
