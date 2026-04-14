#!/usr/bin/env python3
"""
convert_to_webllm.py — Convert a fine-tuned LoRA adapter to WebLLM format and
upload the MLC weights to the Hugging Face Hub.

The conversion pipeline has three stages:

  1. Merge LoRA adapter into the base model weights.
  2. Quantise & convert the merged model using ``mlc_llm`` (MLC-LLM) to produce
     the weight shards and ``mlc-chat-config.json`` that WebLLM requires.
  3. Upload the MLC weight directory to a Hugging Face Hub repository.

Prerequisites::

    pip install mlc-llm huggingface_hub
    # mlc_llm requires a CUDA or Metal GPU for compilation

Usage::

    # Minimal — infers adapter path and base model from training_config.yaml
    python scripts/convert_to_webllm.py \\
        --hub-model-id your-org/mhg-qwen2.5-3b-webllm

    # Explicit paths
    python scripts/convert_to_webllm.py \\
        --adapter-dir  output/mhg-qwen2.5-3b/adapter \\
        --base-model   Qwen/Qwen2.5-3B-Instruct \\
        --merged-dir   output/mhg-qwen2.5-3b/merged \\
        --mlc-dir      output/mhg-qwen2.5-3b/mlc-weights \\
        --quantization q4f16_1 \\
        --hub-model-id your-org/mhg-qwen2.5-3b-webllm

    # Skip re-merging if the merged model already exists
    python scripts/convert_to_webllm.py \\
        --merged-dir output/mhg-qwen2.5-3b/merged \\
        --mlc-dir    output/mhg-qwen2.5-3b/mlc-weights \\
        --hub-model-id your-org/mhg-qwen2.5-3b-webllm

Notes:

* The ``--conv-template`` is inferred automatically from the base-model name
  when not specified (qwen2, llama-3, phi-3, gemma).  Pass it explicitly if
  auto-detection is wrong.
* Only the MLC weight directory is uploaded, not the full merged HuggingFace
  model.  Reuse the prebuilt ``.wasm`` library from the mlc-ai binary repo
  that matches your base-model architecture and quantisation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from rich.console import Console

console = Console()

# Project root: mhg-finetune/ (two levels up from this script)
_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONV_TEMPLATE_MAP: dict[str, str] = {
    "qwen2.5": "qwen2",
    "qwen2":   "qwen2",
    "llama-3": "llama-3",
    "llama-3.2": "llama-3",
    "llama-3.1": "llama-3",
    "phi-3.5":  "phi-3",
    "phi-3":    "phi-3",
    "gemma-2":  "gemma",
    "gemma":    "gemma",
    "mistral":  "mistral",
}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_conv_template(model_id: str) -> str:
    """Return the MLC conv-template name inferred from *model_id*."""
    lower = model_id.lower()
    for key, template in _CONV_TEMPLATE_MAP.items():
        if key in lower:
            return template
    console.print(
        f"[yellow]Warning: could not auto-detect conv-template for '{model_id}'. "
        "Defaulting to 'qwen2'. Pass --conv-template to override.[/yellow]"
    )
    return "qwen2"


def run(cmd: list[str], desc: str) -> None:
    """Run *cmd* as a subprocess, streaming output; raise on non-zero exit."""
    console.print(f"[bold]{desc}[/bold]")
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        console.print(f"[red]Command failed (exit {result.returncode})[/red]")
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def merge_adapter(
    base_model: str,
    adapter_dir: str,
    merged_dir: str,
) -> None:
    """Merge a LoRA adapter into the base model and save the result."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print(f"[bold]Merging adapter into base model…[/bold]")
    console.print(f"  base  : {base_model}")
    console.print(f"  adapter: {adapter_dir}")
    console.print(f"  output : {merged_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    merged = peft_model.merge_and_unload()

    out = Path(merged_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    console.print(f"[green]Merged model saved → {out}[/green]")


def convert_with_mlc(
    merged_dir: str,
    mlc_dir: str,
    quantization: str,
    conv_template: str,
) -> None:
    """Run mlc_llm convert_weight and gen_config on the merged model."""
    mlc_path = Path(mlc_dir)
    mlc_path.mkdir(parents=True, exist_ok=True)

    run(
        [
            "mlc_llm", "convert_weight",
            merged_dir,
            "--quantization", quantization,
            "--output", str(mlc_path),
        ],
        "Converting weights with MLC-LLM…",
    )

    run(
        [
            "mlc_llm", "gen_config",
            merged_dir,
            "--quantization", quantization,
            "--conv-template", conv_template,
            "--output", str(mlc_path),
        ],
        "Generating MLC chat config…",
    )

    console.print(f"[green]MLC weights ready → {mlc_path}[/green]")


def upload_to_hub(mlc_dir: str, hub_model_id: str) -> None:
    """Upload the MLC weight directory to a Hugging Face Hub repository."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        console.print(
            "[red]huggingface_hub is not installed. "
            "Run: pip install huggingface_hub[/red]"
        )
        sys.exit(1)

    console.print(f"[bold]Uploading to Hugging Face Hub: {hub_model_id}[/bold]")
    api = HfApi()
    api.create_repo(repo_id=hub_model_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=mlc_dir,
        repo_id=hub_model_id,
        repo_type="model",
    )
    console.print(
        f"[green]Uploaded → https://huggingface.co/{hub_model_id}[/green]"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        default=str(_ROOT / "configs/training_config.yaml"),
        help="Training config YAML (default: configs/training_config.yaml)",
    )
    parser.add_argument(
        "--base-model",
        help="Base HuggingFace model id or local path. "
             "Defaults to model_name_or_path from the training config.",
    )
    parser.add_argument(
        "--adapter-dir",
        help="Path to the LoRA adapter directory. "
             "Defaults to <output_dir>/adapter from the training config.",
    )
    parser.add_argument(
        "--merged-dir",
        help="Where to save (or load) the merged full model. "
             "Defaults to <output_dir>/merged.",
    )
    parser.add_argument(
        "--mlc-dir",
        help="Where to save the MLC weight shards. "
             "Defaults to <output_dir>/mlc-weights.",
    )
    parser.add_argument(
        "--quantization",
        default="q4f16_1",
        help="MLC quantization preset (default: q4f16_1).",
    )
    parser.add_argument(
        "--conv-template",
        help="MLC conversation template (e.g. qwen2, llama-3, phi-3, gemma). "
             "Auto-detected from the model name when omitted.",
    )
    parser.add_argument(
        "--hub-model-id",
        help="Hugging Face Hub repository id to upload MLC weights to "
             "(e.g. your-org/mhg-qwen2.5-3b-webllm).",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip the merge step (use when --merged-dir already exists).",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip the MLC conversion step (use when --mlc-dir already exists).",
    )
    args = parser.parse_args(argv)

    # ── Resolve paths from config ─────────────────────────────────────────────
    cfg = load_config(args.config)
    _resolve = lambda p: str(_ROOT / p) if not Path(p).is_absolute() else p

    base_model  = args.base_model  or cfg["model_name_or_path"]
    output_dir  = _resolve(cfg.get("output_dir", "output/mhg-model"))
    adapter_dir = args.adapter_dir  or str(Path(output_dir) / "adapter")
    merged_dir  = args.merged_dir   or str(Path(output_dir) / "merged")
    mlc_dir     = args.mlc_dir      or str(Path(output_dir) / "mlc-weights")
    conv_template = args.conv_template or infer_conv_template(base_model)

    console.print("[bold cyan]WebLLM conversion pipeline[/bold cyan]")
    console.print(f"  base model   : {base_model}")
    console.print(f"  adapter      : {adapter_dir}")
    console.print(f"  merged model : {merged_dir}")
    console.print(f"  mlc weights  : {mlc_dir}")
    console.print(f"  quantization : {args.quantization}")
    console.print(f"  conv-template: {conv_template}")
    if args.hub_model_id:
        console.print(f"  hub model id : {args.hub_model_id}")

    # ── Stage 1: Merge ────────────────────────────────────────────────────────
    if args.skip_merge:
        console.print("[yellow]Skipping merge (--skip-merge).[/yellow]")
    else:
        if not Path(adapter_dir).exists():
            console.print(f"[red]Adapter directory not found: {adapter_dir}[/red]")
            sys.exit(1)
        merge_adapter(base_model, adapter_dir, merged_dir)

    # ── Stage 2: MLC conversion ───────────────────────────────────────────────
    if args.skip_convert:
        console.print("[yellow]Skipping MLC conversion (--skip-convert).[/yellow]")
    else:
        if not Path(merged_dir).exists():
            console.print(f"[red]Merged model directory not found: {merged_dir}[/red]")
            sys.exit(1)
        convert_with_mlc(merged_dir, mlc_dir, args.quantization, conv_template)

    # ── Stage 3: Upload ───────────────────────────────────────────────────────
    if args.hub_model_id:
        if not Path(mlc_dir).exists():
            console.print(f"[red]MLC weights directory not found: {mlc_dir}[/red]")
            sys.exit(1)
        upload_to_hub(mlc_dir, args.hub_model_id)
    else:
        console.print(
            "[yellow]No --hub-model-id provided; skipping upload.[/yellow]"
        )

    console.print("\n[bold green]Done.[/bold green]")
    if args.hub_model_id:
        console.print(
            f"\nTo use in WebLLM, point model_lib at the prebuilt .wasm for "
            f"your base architecture + quantization:\n"
            f"  https://github.com/mlc-ai/binary-mlc-llm-libs"
        )


if __name__ == "__main__":
    main()
