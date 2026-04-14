#!/usr/bin/env python3
"""
evaluate.py — Evaluate the fine-tuned MHG model.

Metrics computed:
  • Perplexity on a held-out MHG text set
  • BLEU and chrF on translation examples (MHG → Modern German)
  • MHG heuristic pass-rate on mhg_conversation/paraphrase examples

The script can evaluate:
  (a) A merged full model  (--model-dir path/to/merged)
  (b) A base model + LoRA adapter  (--base-model id --adapter-dir path/to/adapter)

Results are printed to stdout and saved as JSON to output/eval_results.json.

Usage::

    # Evaluate a LoRA adapter on top of the base model
    python scripts/evaluate.py \\
        --base-model Qwen/Qwen2.5-3B-Instruct \\
        --adapter-dir output/mhg-qwen2.5-3b/adapter \\
        --eval-file   data/eval.jsonl \\
        --chunks-file data/chunks.jsonl

    # Evaluate a fully merged model
    python scripts/evaluate.py \\
        --model-dir output/mhg-qwen2.5-3b/merged
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_dir: str | None,
    base_model: str | None,
    adapter_dir: str | None,
):
    if model_dir:
        model_id = model_dir
    elif base_model:
        model_id = base_model
    else:
        raise ValueError("Provide --model-dir or --base-model")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_dir and not model_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """Compute average per-token perplexity over *texts* using a sliding window."""
    device = model.device
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        encodings = tokenizer(text, return_tensors="pt")
        input_ids: torch.Tensor = encodings.input_ids.to(device)
        seq_len = input_ids.size(1)

        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_len = end - prev_end

            with torch.no_grad():
                outputs = model(
                    input_ids[:, begin:end],
                    labels=input_ids[:, begin:end],
                )
            # outputs.loss is mean NLL over all tokens in the window
            total_nll += outputs.loss.item() * target_len
            total_tokens += target_len
            prev_end = end
            if end == seq_len:
                break

    return math.exp(total_nll / total_tokens) if total_tokens else float("inf")


# ---------------------------------------------------------------------------
# Translation quality (BLEU / chrF)
# ---------------------------------------------------------------------------


def compute_translation_metrics(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    max_new_tokens: int = 256,
) -> dict[str, float]:
    """Run the model on translation examples and compute BLEU + chrF."""
    from sacrebleu.metrics import BLEU, CHRF

    device = model.device
    hypotheses: list[str] = []
    references: list[str] = []

    translation_records = [
        r for r in records if r.get("scenario") == "translation_to_modern"
    ]
    if not translation_records:
        return {}

    for record in translation_records:
        messages = record.get("messages", [])
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        ref_msgs  = [m["content"] for m in messages if m["role"] == "assistant"]
        if not user_msgs or not ref_msgs:
            continue

        chat = []
        if record.get("system"):
            chat.append({"role": "system", "content": record["system"]})
        chat.append({"role": "user", "content": user_msgs[0]})

        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        hypothesis = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        hypotheses.append(hypothesis)
        references.append(ref_msgs[0].strip())

    if not hypotheses:
        return {}

    bleu  = BLEU(effective_order=True)
    chrf  = CHRF()
    bleu_score = bleu.corpus_score(hypotheses, [references]).score
    chrf_score = chrf.corpus_score(hypotheses, [references]).score
    return {"bleu": bleu_score, "chrf": chrf_score, "n_translation": len(hypotheses)}


# ---------------------------------------------------------------------------
# MHG heuristic pass-rate
# ---------------------------------------------------------------------------


def compute_mhg_pass_rate(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    max_new_tokens: int = 256,
) -> dict[str, float]:
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.mhg_heuristics import is_likely_mhg

    device = model.device
    mhg_records = [
        r for r in records
        if r.get("scenario") in {"mhg_conversation", "paraphrase"}
    ]
    if not mhg_records:
        return {}

    passed = 0
    for record in mhg_records:
        messages = record.get("messages", [])
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        if not user_msgs:
            continue

        chat = []
        if record.get("system"):
            chat.append({"role": "system", "content": record["system"]})
        chat.append({"role": "user", "content": user_msgs[0]})

        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if is_likely_mhg(response):
            passed += 1

    n = len(mhg_records)
    return {"mhg_pass_rate": passed / n if n else 0.0, "n_mhg": n}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",    help="Path to a fully merged model directory")
    parser.add_argument("--base-model",   help="Base model HF id or path")
    parser.add_argument("--adapter-dir",  help="LoRA adapter directory")
    parser.add_argument("--eval-file",    default="data/eval.jsonl")
    parser.add_argument("--chunks-file",  default="data/chunks.jsonl",
                        help="Raw MHG chunks for perplexity evaluation")
    parser.add_argument("--output",       default="output/eval_results.json")
    parser.add_argument(
        "--perplexity-samples", type=int, default=50,
        help="Number of raw MHG chunks to use for perplexity (default: 50)",
    )
    parser.add_argument(
        "--translation-samples", type=int, default=100,
        help="Max translation examples for BLEU/chrF (default: 100)",
    )
    parser.add_argument(
        "--mhg-samples", type=int, default=100,
        help="Max MHG-output examples for pass-rate (default: 100)",
    )
    args = parser.parse_args(argv)

    # ── Load model ────────────────────────────────────────────────────────────
    console.print("[bold]Loading model…[/bold]")
    model, tokenizer = load_model_and_tokenizer(
        args.model_dir, args.base_model, args.adapter_dir
    )

    results: dict[str, Any] = {}

    # ── Perplexity ────────────────────────────────────────────────────────────
    chunks_path = Path(args.chunks_file)
    if chunks_path.exists():
        console.print("[bold]Computing perplexity…[/bold]")
        chunks: list[dict] = []
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        import random
        random.seed(42)
        sample = random.sample(chunks, min(args.perplexity_samples, len(chunks)))
        texts = [c["text"] for c in sample]
        ppl = compute_perplexity(model, tokenizer, texts)
        results["perplexity"] = ppl
        console.print(f"  Perplexity: [green]{ppl:.2f}[/green]")
    else:
        console.print(f"[yellow]Chunks file not found ({chunks_path}), skipping perplexity.[/yellow]")

    # ── Load eval records ─────────────────────────────────────────────────────
    eval_path = Path(args.eval_file)
    eval_records: list[dict] = []
    if eval_path.exists():
        with eval_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_records.append(json.loads(line))
    else:
        console.print(f"[yellow]Eval file not found ({eval_path}), skipping downstream metrics.[/yellow]")

    # ── Translation ───────────────────────────────────────────────────────────
    if eval_records:
        import random
        random.seed(42)
        sample = random.sample(eval_records, min(args.translation_samples, len(eval_records)))
        console.print("[bold]Computing translation metrics…[/bold]")
        t_metrics = compute_translation_metrics(model, tokenizer, sample)
        results.update(t_metrics)
        if t_metrics:
            console.print(
                f"  BLEU: [green]{t_metrics.get('bleu', 0):.1f}[/green]  "
                f"chrF: [green]{t_metrics.get('chrf', 0):.1f}[/green]  "
                f"(n={t_metrics.get('n_translation', 0)})"
            )

    # ── MHG pass-rate ─────────────────────────────────────────────────────────
    if eval_records:
        import random
        random.seed(42)
        sample = random.sample(eval_records, min(args.mhg_samples, len(eval_records)))
        console.print("[bold]Computing MHG heuristic pass-rate…[/bold]")
        m_metrics = compute_mhg_pass_rate(model, tokenizer, sample)
        results.update(m_metrics)
        if m_metrics:
            pct = m_metrics.get("mhg_pass_rate", 0) * 100
            console.print(
                f"  MHG pass-rate: [green]{pct:.1f}%[/green]  "
                f"(n={m_metrics.get('n_mhg', 0)})"
            )

    # ── Summary table ─────────────────────────────────────────────────────────
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="green")
    for k, v in results.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
