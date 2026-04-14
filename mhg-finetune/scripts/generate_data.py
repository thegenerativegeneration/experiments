#!/usr/bin/env python3
"""
generate_data.py — Generate synthetic MHG chat pairs using a powerful LLM.

For each text chunk in data/chunks.jsonl the script:
  1. Samples a scenario (mhg_conversation, translation, explanation, etc.)
  2. Builds a prompt that includes the MHG chunk as context
  3. Calls the configured LLM API (OpenAI or Anthropic)
  4. Writes the result as a chat record to data/synthetic_raw.jsonl

Each output record::

    {
        "source": "nibelungenlied",
        "chunk_id": 42,
        "scenario": "mhg_conversation",
        "system": "Du bist ein Muttersprachler ...",
        "messages": [
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "context_text": "Uns ist in alten mæren ..."
    }

Usage::

    # OpenAI (set OPENAI_API_KEY)
    python scripts/generate_data.py --config configs/generation_config.yaml

    # Anthropic (set ANTHROPIC_API_KEY)
    python scripts/generate_data.py --config configs/generation_config.yaml \\
        --provider anthropic --model claude-3-5-sonnet-20241022

    # Dry-run: print prompts without calling the API
    python scripts/generate_data.py --dry-run --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# ---------------------------------------------------------------------------
# User-turn templates per scenario
# ---------------------------------------------------------------------------

USER_TEMPLATES: dict[str, list[str]] = {
    "mhg_conversation": [
        "Waz ist dîn name und von wannen bist dû?",
        "Erzele mir von dînem leben und von dînem herren.",
        "Wâ bist dû gestern gewesen, und waz hâstu getân?",
        "Wie gefelt dir diu minne der vrouwen in disem lande?",
        "Sage mir: waz ist daz edelste dinc in dirre welt?",
        "Wie heizet der recke, der in dem gedichte ist genant?",
        "Von welchem lande koment die helden in disem mære?",
    ],
    "translation_to_modern": [
        "Translate this Middle High German passage into modern German:\n\n{text}",
        "Please provide a modern German translation of the following MHG text:\n\n{text}",
        "Wie lautet dieser mittelhochdeutsche Text auf Neuhochdeutsch?\n\n{text}",
    ],
    "explanation": [
        "Explain this Middle High German passage in English, "
        "covering vocabulary, grammar, and literary context:\n\n{text}",
        "What does this MHG text mean, and what grammatical features are notable?\n\n{text}",
        "Analyse the following Middle High German excerpt — meaning, grammar, style:\n\n{text}",
    ],
    "grammar_qa": [
        "In the passage above, identify all dative case forms and explain their function.",
        "What verb forms appear in this text, and how do they differ from Modern German?",
        "Explain the pronoun system illustrated by this passage.",
        "How does word order differ from Modern German in this excerpt?",
        "List the adjective inflections you can find and describe the inflection class.",
        "What are the notable features of the verb conjugation in this passage?",
    ],
    "paraphrase": [
        "Formuliere diesen mittelhochdeutschen Abschnitt in einem anderen Stil um:\n\n{text}",
        "Rewrite this MHG passage using different vocabulary while keeping the meaning:\n\n{text}",
    ],
}


def _sample_user_turn(scenario: str, chunk_text: str) -> str:
    templates = USER_TEMPLATES[scenario]
    template = random.choice(templates)
    return template.format(text=chunk_text)


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------


def _call_openai(
    system: str,
    user: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    from openai import OpenAI  # lazy import
    client = OpenAI()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
    def _call() -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    return _call()


def _call_anthropic(
    system: str,
    user: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    import anthropic  # lazy import
    client = anthropic.Anthropic()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
    def _call() -> str:
        response = client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.content[0].text

    return _call()


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _weighted_choice(scenario_weights: dict[str, float]) -> str:
    scenarios = list(scenario_weights.keys())
    weights = [scenario_weights[s] for s in scenarios]
    return random.choices(scenarios, weights=weights, k=1)[0]


def _build_system_prompt(scenario: str, system_prompts: dict[str, str]) -> str:
    return system_prompts.get(scenario, "You are a helpful assistant.")


def generate_examples(
    chunk: dict[str, Any],
    n: int,
    config: dict[str, Any],
    provider: str,
    model: str,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    scenario_weights: dict[str, float] = config["scenario_weights"]
    system_prompts: dict[str, str] = config["system_prompts"]
    max_tokens: int = config.get("max_tokens", 1024)
    temperature: float = config.get("temperature", 0.8)
    chunk_text: str = chunk["text"]

    for _ in range(n):
        scenario = _weighted_choice(scenario_weights)

        # For grammar_qa we prepend the chunk to the system prompt
        if scenario == "grammar_qa":
            system = (
                system_prompts["grammar_qa"]
                + f"\n\nText under discussion:\n{chunk_text}"
            )
            user = _sample_user_turn("grammar_qa", chunk_text)
        else:
            system = _build_system_prompt(scenario, system_prompts)
            user = _sample_user_turn(scenario, chunk_text)

        if dry_run:
            assistant = f"[DRY RUN — {scenario}]"
        else:
            try:
                if provider == "openai":
                    assistant = _call_openai(system, user, model, max_tokens, temperature)
                elif provider == "anthropic":
                    assistant = _call_anthropic(system, user, model, max_tokens, temperature)
                else:
                    raise ValueError(f"Unknown provider: {provider!r}")
            except Exception as exc:  # noqa: BLE001
                tqdm.write(f"  WARNING: API error for chunk {chunk.get('chunk_id')}: {exc}")
                continue

        record: dict[str, Any] = {
            "source": chunk.get("source", ""),
            "chunk_id": chunk.get("chunk_id", -1),
            "scenario": scenario,
            "system": system,
            "messages": [
                {"role": "user",      "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "context_text": chunk_text,
        }
        results.append(record)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/generation_config.yaml",
        help="Path to generation_config.yaml",
    )
    parser.add_argument("--provider", help="Override config provider (openai|anthropic)")
    parser.add_argument("--model",    help="Override config model name")
    parser.add_argument("--chunks-file", help="Override config chunks_file path")
    parser.add_argument("--output",      help="Override config output_file path")
    parser.add_argument(
        "--examples-per-chunk", type=int,
        help="Override examples_per_chunk",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only the first N chunks (0 = all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build prompts but do not call the API",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    random.seed(args.seed)

    config = _load_config(args.config)
    provider: str = args.provider or config["provider"]
    model: str    = args.model    or config["model"]
    chunks_file   = Path(args.chunks_file or config["chunks_file"])
    output_file   = Path(args.output      or config["output_file"])
    n_per_chunk   = args.examples_per_chunk or config.get("examples_per_chunk", 4)

    if not chunks_file.exists():
        print(f"ERROR: chunks file not found: {chunks_file}", file=sys.stderr)
        print("Run scripts/preprocess.py first.", file=sys.stderr)
        sys.exit(1)

    # Load chunks
    chunks: list[dict[str, Any]] = []
    with chunks_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if args.limit:
        chunks = chunks[: args.limit]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    with output_file.open("w", encoding="utf-8") as fout:
        for chunk in tqdm(chunks, desc="Generating"):
            examples = generate_examples(
                chunk, n_per_chunk, config, provider, model, args.dry_run
            )
            for ex in examples:
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            total_generated += len(examples)
            # Brief pause between chunks to respect rate limits
            if not args.dry_run:
                time.sleep(0.5)

    print(f"\nGenerated {total_generated} examples → {output_file}")


if __name__ == "__main__":
    main()
