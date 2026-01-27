# cross-lingual-stability-judges
2026 EACL MME Workshop: Cross-Lingual Stability of LLM Judges Under Controlled Generation: Evidence from Finno-Ugric Languages

## Setup

```bash
pip install -e .
cp .env.example .env  # then add your OPENAI_API_KEY
```

## Step 1: Generate Conversations

Generate synthetic customer support conversations in Finno-Ugric languages (Estonian, Finnish, Hungarian) and English.

```bash
python -m conversation_generator -n 100 -l et  # 100 Estonian conversations
python -m conversation_generator -n 100 -l fi  # 100 Finnish conversations
python -m conversation_generator -n 100 -l hu  # 100 Hungarian conversations
python -m conversation_generator --help        # all options
```

Output files are named `{model}_{lang}_{datetime}.jsonl` (e.g., `gpt-4.1-mini_et_20260127-143052.jsonl`).

## Step 2: Combine and Validate

Combine JSONL files from multiple generation runs into a single JSON file grouped by model. The combined output is the central file used for all downstream analyses.

```bash
# Combine all languages
python -m conversation_parser data/*.jsonl --stats

# Combine single language files
python -m conversation_parser data/*_et_*.jsonl --stats
```

Output is JSON grouped by model:
```json
{
  "gpt-4.1-mini": [
    {"subject": "...", "messages": [...], "source_file": "...", "_metadata": {...}},
    ...
  ],
  "gpt-4": [...]
}
```

Output files are named `combined_{lang}.json`:
- Single language input → `combined_et.json`
- Multiple languages → `combined_mixed.json`

Use `--stats` alone to validate without writing output:
```bash
python -m conversation_parser data/*.jsonl --stats
```
