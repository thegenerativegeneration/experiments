#!/usr/bin/env python3
"""
train.py — QLoRA supervised fine-tuning for the MHG conversational model.

Reads a YAML training config (default: configs/training_config.yaml) and
fine-tunes the specified base model on data/train.jsonl using HuggingFace TRL's
SFTTrainer with 4-bit quantisation (QLoRA / bitsandbytes).

The fine-tuned LoRA adapter (and optionally a merged model) is saved to
output/<run_name>/.

Usage::

    # Basic run (reads configs/training_config.yaml)
    python scripts/train.py

    # Override the base model and output dir
    python scripts/train.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --output-dir output/mhg-qwen1.5b

    # Push merged model to Hugging Face Hub after training
    python scripts/train.py --push-to-hub --hub-model-id your-org/mhg-qwen2.5-3b
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_bnb_config(cfg: dict) -> BitsAndBytesConfig | None:
    if not cfg.get("load_in_4bit", True):
        return None
    dtype_str = cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
    )


def build_lora_config(cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=cfg.get("lora_r", 32),
        lora_alpha=cfg.get("lora_alpha", 64),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        ),
        bias=cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )


def format_chat(record: dict, tokenizer) -> str:
    """Convert a data record to a single string using the model's chat template."""
    messages = []
    if record.get("system"):
        messages.append({"role": "system", "content": record["system"]})
    messages.extend(record.get("messages", []))
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training_config.yaml",
        help="Path to training_config.yaml",
    )
    parser.add_argument("--model",      help="Override model_name_or_path")
    parser.add_argument("--train-file", help="Override train_file path")
    parser.add_argument("--eval-file",  help="Override eval_file path")
    parser.add_argument("--output-dir", help="Override output_dir path")
    parser.add_argument("--epochs",     type=int, help="Override num_train_epochs")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", help="Hub repo id for push")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    # CLI overrides
    model_id    = args.model      or cfg["model_name_or_path"]
    train_file  = args.train_file or cfg.get("train_file", "data/train.jsonl")
    eval_file   = args.eval_file  or cfg.get("eval_file",  "data/eval.jsonl")
    output_dir  = args.output_dir or cfg.get("output_dir", "output/mhg-model")
    num_epochs  = args.epochs     or cfg.get("num_train_epochs", 3)
    push_to_hub = args.push_to_hub or cfg.get("push_to_hub", False)
    hub_model_id = args.hub_model_id or cfg.get("hub_model_id", "")

    # ── Load tokeniser ────────────────────────────────────────────────────────
    print(f"Loading tokeniser: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset: {train_file} / {eval_file}")
    raw = load_dataset(
        "json",
        data_files={"train": train_file, "eval": eval_file},
    )

    def _format_record(record):
        return {"text": format_chat(record, tokenizer)}

    train_ds = raw["train"].map(_format_record, remove_columns=raw["train"].column_names)
    eval_ds  = raw["eval"].map(_format_record,  remove_columns=raw["eval"].column_names)

    # ── Load model ────────────────────────────────────────────────────────────
    bnb_config = build_bnb_config(cfg)
    print(f"Loading model: {model_id} (4-bit={bnb_config is not None})")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.get("bf16") else torch.float16,
    )
    model.config.use_cache = False

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_config = build_lora_config(cfg)

    # ── Training arguments ────────────────────────────────────────────────────
    use_bf16 = cfg.get("bf16", True)
    use_fp16 = cfg.get("fp16", False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        optim=cfg.get("optim", "paged_adamw_32bit"),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        save_steps=cfg.get("save_steps", 100),
        eval_steps=cfg.get("eval_steps", 100),
        eval_strategy="steps",
        logging_steps=cfg.get("logging_steps", 25),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        seed=cfg.get("seed", 42),
        report_to="none",
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id or None,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=cfg.get("max_seq_length", 2048),
        packing=cfg.get("packing", True),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Starting training…")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    adapter_path = Path(output_dir) / "adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"LoRA adapter saved → {adapter_path}")

    if push_to_hub and hub_model_id:
        trainer.push_to_hub(hub_model_id)
        print(f"Model pushed to Hub: {hub_model_id}")


if __name__ == "__main__":
    main()
