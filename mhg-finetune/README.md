# MHG Fine-Tune

Fine-tuning a small LLM (1.6B–5B parameters) for **Middle High German
(Mittelhochdeutsch)** conversational AI, using synthetic chat data generated
from public-domain MHG source texts.

> **Coming next:** a browser-based chat demo powered by the fine-tuned model
> running entirely in-browser via [WebLLM](https://webllm.mlc.ai/) (WebGPU).
> The model is intentionally kept in the 1.6B–5B range so it can be quantised
> and served directly in the browser.

---

## What this experiment does

| Step | Script | Description |
|------|--------|-------------|
| 1 | `collect_texts.py` | Fetch MHDBDB TEI files from GitHub |
| 2 | `preprocess.py` | Clean and chunk texts into ~350-token overlapping passages |
| 3 | `generate_data.py` | Use GPT-4o / Claude to generate synthetic chat pairs from the passages |
| 4 | `clean_data.py` | Filter, validate MHG authenticity, deduplicate, and split train/eval |
| 5 | `train.py` | QLoRA SFT fine-tuning with HuggingFace TRL |
| 6 | `evaluate.py` | Perplexity, BLEU/chrF, and MHG heuristic pass-rate |

---

## Conversational modes

The generated dataset covers five complementary scenarios:

| Scenario | System role | User turn | Assistant turn |
|----------|-------------|-----------|----------------|
| `mhg_conversation` | Native MHG speaker (~1200) | MHG question | MHG answer |
| `translation_to_modern` | MHG→Modern German translator | MHG passage | Modern German |
| `explanation` | MHG scholar | MHG passage | English explanation |
| `grammar_qa` | MHG grammar tutor | Grammar question | MHG-grounded answer |
| `paraphrase` | Medieval poet | MHG passage | MHG paraphrase |

---

## Recommended base models (WebLLM-compatible)

| Model | Params | Notes |
|-------|--------|-------|
| **Qwen/Qwen2.5-3B-Instruct** *(default)* | 3B | Strong multilingual, WebLLM support |
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | Lightest option |
| meta-llama/Llama-3.2-3B-Instruct | 3B | Good German, WebLLM support |
| microsoft/Phi-3.5-mini-instruct | 3.8B | Strong reasoning |
| google/gemma-2-2b-it | 2B | Strong European language coverage |

All of the above have pre-built MLC-LLM configs and can be served in the
browser after quantisation.

---

## Quick start

### 1. Install dependencies

```bash
cd mhg-finetune
pip install -r requirements.txt
```

### 2. Collect MHG source texts

```bash
python scripts/collect_texts.py
# → data/raw/*.txt
```

### 3. Chunk the texts

```bash
python scripts/preprocess.py
# → data/chunks.jsonl
```

### 4. Generate synthetic chat data

```bash
export OPENAI_API_KEY=sk-...

python scripts/generate_data.py \
    --config configs/generation_config.yaml
# → data/synthetic_raw.jsonl   (~5 000 examples at default settings)
```

To use Anthropic Claude instead of GPT-4o:

```bash
export ANTHROPIC_API_KEY=sk-ant-...

python scripts/generate_data.py \
    --provider anthropic \
    --model claude-3-5-sonnet-20241022
```

### 5. Clean and split the data

```bash
python scripts/clean_data.py
# → data/train.jsonl  (90 %)
# → data/eval.jsonl   (10 %)
```

### 6. Fine-tune

```bash
# Requires a CUDA GPU; single A100 / 2× RTX 4090 recommended
python scripts/train.py
# → output/mhg-qwen2.5-3b/adapter/
```

To use a different base model:

```bash
python scripts/train.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir output/mhg-qwen1.5b
```

### 7. Evaluate

```bash
python scripts/evaluate.py \
    --base-model  Qwen/Qwen2.5-3B-Instruct \
    --adapter-dir output/mhg-qwen2.5-3b/adapter \
    --eval-file   data/eval.jsonl \
    --chunks-file data/chunks.jsonl
# → output/eval_results.json
```

---

## Directory structure

```
mhg-finetune/
├── requirements.txt
├── configs/
│   ├── generation_config.yaml   # LLM data-generation settings
│   └── training_config.yaml     # QLoRA fine-tuning hyperparameters
├── data/                        # populated by scripts (gitignored)
│   ├── raw/                     # downloaded source texts
│   ├── chunks.jsonl             # preprocessed passages
│   ├── synthetic_raw.jsonl      # raw generated chat pairs
│   ├── train.jsonl              # filtered training set
│   └── eval.jsonl               # filtered evaluation set
├── output/                      # model checkpoints (gitignored)
└── scripts/
    ├── collect_texts.py
    ├── preprocess.py
    ├── generate_data.py
    ├── clean_data.py
    ├── train.py
    ├── evaluate.py
    └── utils/
        ├── mhg_heuristics.py    # MHG language-detection helpers
        └── dedup.py             # MinHash deduplication
```

---

## Configuration

### `configs/generation_config.yaml`

Key settings:

```yaml
provider: openai          # "openai" or "anthropic"
model: gpt-4o
examples_per_chunk: 4     # chat examples generated per text passage
temperature: 0.8
scenario_weights:
  mhg_conversation:    0.30
  translation_to_modern: 0.25
  explanation:         0.20
  grammar_qa:          0.15
  paraphrase:          0.10
```

### `configs/training_config.yaml`

Key settings:

```yaml
model_name_or_path: Qwen/Qwen2.5-3B-Instruct
lora_r: 32
lora_alpha: 64
num_train_epochs: 3
learning_rate: 2.0e-4
max_seq_length: 2048
```

---

## Data quality

`clean_data.py` runs four automated filters before deduplication:

1. **Schema** — record must have a system prompt and at least one user/assistant turn
2. **Length** — assistant response ≥ 30 characters
3. **MHG heuristics** — for MHG-output scenarios, the response must contain
   characteristic MHG function words and stay below the Modern German contamination
   threshold (see `scripts/utils/mhg_heuristics.py`)
4. **Back-translation** (optional, requires `OPENAI_API_KEY`) — generated MHG
   is translated back to Modern German and compared to the original passage
   using chrF; examples below the threshold are discarded

---

## Roadmap

- [x] Data collection pipeline
- [x] Synthetic data generation (multi-scenario)
- [x] Data cleaning & deduplication
- [x] QLoRA fine-tuning script
- [x] Evaluation (perplexity, BLEU, chrF, MHG pass-rate)
- [ ] DPO preference data generation and training (Phase 2)
- [ ] WebLLM in-browser demo
- [ ] Published dataset on Hugging Face
- [ ] Published model on Hugging Face

---

## Source texts

All source texts are public-domain works fetched from Project Gutenberg and
Wikisource:

| File | Work | Author |
|------|------|--------|
| `nibelungenlied.txt` | Das Nibelungenlied | Anonymous (~1200) |
| `parzival_wolfram.txt` | Parzival | Wolfram von Eschenbach (~1210) |
| `tristan_gottfried.txt` | Tristan | Gottfried von Strassburg (~1210) |
| `iwein_hartmann.txt` | Iwein | Hartmann von Aue (~1200) |
| `arme_heinrich.txt` | Der arme Heinrich | Hartmann von Aue (~1195) |
| `walther_lieder.txt` | Lyric poems | Walther von der Vogelweide (~1190–1230) |
| `minnesang_fruehling.txt` | Minnesangs Frühling | Various (~1150–1200) |
